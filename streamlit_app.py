"""
streamlit_app.py — MediBot Streamlit frontend

Architecture:
  - This script launches backend.py (FastAPI) as a background daemon thread on
    127.0.0.1:8000, then polls /health until the RAG pipeline is ready.
  - All agent calls are made over HTTP via httpx — this file never imports
    anything from src/. The full RAG stack lives exclusively in backend.py.
  - Streamlit itself serves the UI on 0.0.0.0:7860 (the port HF Spaces exposes).

Thread-start guard:
  Module-level code in Streamlit re-runs on every user interaction. To avoid
  re-launching uvicorn on each rerun, _start_fastapi_if_needed() checks whether
  port 8000 is already bound before starting a new thread.
"""

import re
import socket
import threading
import time

import httpx
import streamlit as st
import uvicorn
from audio_recorder_streamlit import audio_recorder


# ── FastAPI background server ─────────────────────────────────────────────────────

def _port_is_bound(host: str, port: int) -> bool:
    """Return True if something is already listening on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.1)
        return s.connect_ex((host, port)) == 0


def _start_fastapi_if_needed() -> None:
    """
    Launch FastAPI (backend.py) on 127.0.0.1:8000 as a daemon thread.
    The port check makes this call idempotent — safe to call on every Streamlit rerun.
    Daemon=True ensures the thread dies automatically when the main process exits.
    """
    if _port_is_bound("127.0.0.1", 8000):
        return  # Already running — skip
    thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": "backend:app",
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "warning",
        },
        daemon=True,
    )
    thread.start()


# Called at module level — runs on every Streamlit rerun, but is idempotent.
_start_fastapi_if_needed()


# ── httpx client helpers ──────────────────────────────────────────────────────────
# All communication with FastAPI goes through these five functions.
# Streamlit's main thread is synchronous, so we use httpx (sync), not asyncio.

BACKEND = "http://127.0.0.1:8000"

# Timeouts: agent + vision calls can be slow; TTS and transcription are faster.
_LONG  = httpx.Timeout(120.0)
_MED   = httpx.Timeout(60.0)
_SHORT = httpx.Timeout(10.0)


def _call_chat(message: str) -> dict:
    """POST /get — text-only message. Returns {response, status_updates}."""
    r = httpx.post(f"{BACKEND}/get", json={"msg": message}, timeout=_LONG)
    r.raise_for_status()
    return r.json()


def _call_analyze_image(message: str, image_bytes: bytes, filename: str) -> dict:
    """POST /analyze-image — image bytes + optional question. Multipart form."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    r = httpx.post(
        f"{BACKEND}/analyze-image",
        files={"image": (filename, image_bytes, mime)},
        data={"question": message},
        timeout=_LONG,
    )
    r.raise_for_status()
    return r.json()


def _call_transcribe(audio_bytes: bytes) -> str:
    """POST /transcribe — WAV bytes from audio-recorder-streamlit → transcribed text."""
    r = httpx.post(
        f"{BACKEND}/transcribe",
        files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
        timeout=_MED,
    )
    r.raise_for_status()
    return r.json().get("text", "")


def _call_speak(text: str) -> bytes:
    """POST /speak — plain text → raw MP3 bytes for st.audio(). Not JSON."""
    r = httpx.post(f"{BACKEND}/speak", json={"text": text}, timeout=_MED)
    r.raise_for_status()
    return r.content  # FileResponse — raw bytes, not .json()


def _call_reset_memory() -> None:
    """POST /reset-memory — clears server-side conversation memory."""
    try:
        httpx.post(f"{BACKEND}/reset-memory", timeout=_SHORT)
    except Exception:
        pass  # Backend may still be starting — not critical at session init


def _wait_for_backend(max_wait: int = 60) -> bool:
    """Poll /health once per second until FastAPI responds 200 or we time out."""
    for _ in range(max_wait):
        try:
            r = httpx.get(f"{BACKEND}/health", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ── Page config (must be the first Streamlit call in the script) ──────────────────

st.set_page_config(page_title="MediBot", page_icon="🩺", layout="wide")


# ── Backend health check — once per browser session ───────────────────────────────
# `backend_ready` is stored in session_state so the spinner only shows on the
# very first load of a new session, not on every subsequent user interaction.

if "backend_ready" not in st.session_state:
    with st.spinner("Starting MediBot backend… (first load takes ~20–30 s while the RAG pipeline initializes)"):
        ready = _wait_for_backend(max_wait=60)
    if not ready:
        st.error(
            "⚠️ Backend failed to start within 60 seconds. "
            "Check terminal logs for errors (missing API keys, Pinecone connection, etc.)."
        )
        st.stop()
    st.session_state.backend_ready = True


# ── Session state — initialized once per browser session ─────────────────────────

if "initialized" not in st.session_state:
    st.session_state.initialized       = True
    st.session_state.messages          = []     # [{"role": "user"|"assistant", "content": str}]
    st.session_state.pending_image_bytes = None # bytes of attached image, or None
    st.session_state.pending_image_name  = None # original filename, for extension + MIME detection
    st.session_state.last_bot_message    = None # raw text of last bot reply, for TTS
    st.session_state.pending_voice_message = None # transcribed text to auto-submit as next message
    st.session_state.last_audio_hash     = None # dedup: skip re-transcribing same recording
    st.session_state.file_uploader_key   = 0    # increment to programmatically reset the uploader
    _call_reset_memory()                         # fresh server-side memory for this session


# ── Sidebar ───────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("MediBot 🩺")
    st.caption("Medical Q&A grounded in the Gale Encyclopedia of Medicine.")
    st.divider()

    # ── Image upload ──────────────────────────────────────────────────────────────
    st.subheader("📷 Attach an Image")
    st.caption("Upload a symptom photo (JPG, PNG, WebP). Sends with your next message.")

    uploaded_file = st.file_uploader(
        "Upload symptom image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.file_uploader_key}",
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()  # reads without advancing the pointer
        st.session_state.pending_image_bytes = image_bytes
        st.session_state.pending_image_name  = uploaded_file.name
        st.image(image_bytes, caption="Attached — sends with your next message.", use_container_width=True)
    elif st.session_state.pending_image_bytes is not None:
        # User removed the file from the uploader widget — clear pending state
        st.session_state.pending_image_bytes = None
        st.session_state.pending_image_name  = None

    st.divider()

    # ── Voice input ───────────────────────────────────────────────────────────────
    st.subheader("🎙️ Voice Input")
    st.caption("Click the mic, speak, click again to stop. Message sends automatically.")

    audio_bytes = audio_recorder(
        text="",
        recording_color="#e53e3e",
        neutral_color="#6c757d",
        icon_name="microphone",
        icon_size="2x",
    )

    if audio_bytes:
        audio_hash = hash(audio_bytes)
        if audio_hash != st.session_state.last_audio_hash:
            # New recording — transcribe and queue for auto-submit in the main area
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("Transcribing…"):
                try:
                    text = _call_transcribe(audio_bytes)
                    st.session_state.pending_voice_message = text
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    st.session_state.pending_voice_message = None

    st.divider()

    # ── Voice output ──────────────────────────────────────────────────────────────
    st.subheader("🔊 Voice Output")
    if st.session_state.last_bot_message:
        if st.button("🔊 Read last response", use_container_width=True):
            with st.spinner("Generating audio…"):
                try:
                    # Light markdown strip so gTTS doesn't read asterisks/backticks aloud
                    clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', st.session_state.last_bot_message)
                    clean = re.sub(r'[*_`#>~]', '', clean).strip()
                    mp3_bytes = _call_speak(clean)
                    st.audio(mp3_bytes, format="audio/mp3", autoplay=True)
                except Exception as e:
                    st.error(f"Voice output failed: {e}")
    else:
        st.caption("Send a message to enable voice output.")

    st.divider()

    # ── Clear chat ────────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages            = []
        st.session_state.last_bot_message    = None
        st.session_state.pending_image_bytes = None
        st.session_state.pending_image_name  = None
        st.session_state.pending_voice_message = None
        st.session_state.last_audio_hash     = None
        st.session_state.file_uploader_key  += 1  # resets the file uploader widget
        _call_reset_memory()
        st.rerun()

    st.divider()
    st.caption("⚠️ Educational use only. Not a substitute for professional medical advice.")


# ── Main chat area ────────────────────────────────────────────────────────────────

st.header("MediBot AI 🩺")
st.caption("Ask about symptoms, treatments, or medical conditions.")

# Render full conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def _process_message(user_input: str) -> None:
    """Show the user message, call FastAPI, render the bot response.
    Shared by both the chat_input widget and the voice auto-submit path."""

    # 1. Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Call the appropriate FastAPI endpoint
    with st.spinner("🧠 MediBot is thinking…"):
        try:
            if st.session_state.pending_image_bytes:
                data = _call_analyze_image(
                    message=user_input,
                    image_bytes=st.session_state.pending_image_bytes,
                    filename=st.session_state.pending_image_name or "image.jpg",
                )
            else:
                data = _call_chat(user_input)
            answer = data.get("response") or "I'm sorry, I couldn't generate a response. Please try again."
        except httpx.HTTPStatusError as e:
            answer = f"Backend error ({e.response.status_code}): {e.response.text}"
        except Exception as e:
            answer = f"Something went wrong: {e}"
        finally:
            st.session_state.pending_image_bytes = None
            st.session_state.pending_image_name  = None
            st.session_state.file_uploader_key  += 1

    # 3. Show and store the bot response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_bot_message = answer
    with st.chat_message("assistant"):
        st.markdown(answer)


# Voice auto-submit — fires on the same rerun that transcription completed.
# The sidebar sets pending_voice_message, then script execution continues here.
if st.session_state.pending_voice_message:
    voice_msg = st.session_state.pending_voice_message
    st.session_state.pending_voice_message = None
    _process_message(voice_msg)

# Regular text input — always rendered at the bottom of the main area
if user_input := st.chat_input("Type your medical query…"):
    _process_message(user_input)
