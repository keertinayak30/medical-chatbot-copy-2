---
title: Medical Chatbot V2
emoji: 👩🏻‍⚕️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
---

# MediBot AI: Multimodal Agentic Medical Assistant

**Live Demo:** [Launch MediBot V2](https://keertinayak30-medical-chatbot-v2.hf.space/)

## Overview

MediBot is an end-to-end multimodal medical Q&A chatbot grounded in the Gale Encyclopedia of Medicine. Users can interact via text, voice, or by uploading an image of a visible symptom (such as a skin condition), and the system returns a grounded answer retrieved from a vector database of medical literature. On top of the core RAG pipeline, MediBot uses an agentic LangGraph layer that classifies intent, asks clarifying questions when input is vague, and triggers safety warnings for medical emergencies.

---

## Features

| Feature | Description |
|---|---|
| **RAG** | Answers grounded in the Gale Encyclopedia of Medicine, indexed in Pinecone |
| **Voice Input** | Speak symptoms; transcribed via Whisper Large V3 on Groq |
| **Image Analysis** | Upload photos of visible symptoms; vision model describes, then RAG retrieves |
| **Voice Output** | Bot responses read aloud via Google Text-to-Speech at 1.25x speed |
| **Agentic Reasoning** | 7-node LangGraph state machine routes intent: chat, clarify, retrieve, or emergency |
| **Urgency Detection** | Screens every message for red flags; prepends emergency warning when triggered |
| **Symptom Clarification** | Asks a focused follow-up for vague input instead of hallucinating |
| **Conversational Memory** | Per-session window of last 5 exchanges for natural follow-ups |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit (`st.chat_message`, `st.chat_input`, `st.file_uploader`, `st.audio`, `audio-recorder-streamlit`) |
| **Backend API** | FastAPI + uvicorn (runs as background daemon thread on `127.0.0.1:8000`) |
| **LLM Orchestration** | LangChain |
| **Agent Framework** | LangGraph (conditional state machine) |
| **LLM** | Llama 3.3 70B Versatile via Groq Cloud |
| **Vision Model** | Llama 4 Scout 17B via Groq Cloud |
| **Speech-to-Text** | Whisper Large V3 via Groq Cloud |
| **Text-to-Speech** | Google Text-to-Speech (gTTS) |
| **Vector Database** | Pinecone (serverless, cosine similarity, 384 dimensions) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Deployment** | Docker on Hugging Face Spaces + GitHub Actions CI/CD |

---

## Architecture

### Request Flow

**Text query**
User input → FastAPI `/get` → LangGraph agent → urgency check → intent classification → direct chat / clarification / RAG retrieval → Pinecone vector search → Llama 3.3 70B response → assembled with safety warnings if applicable → returned to Streamlit frontend.

**Voice query**
In-browser mic capture → WAV bytes → POST to `/transcribe` → Whisper transcription → text inserted into chat input → same flow as text query.

**Image query**
Image upload → POST to `/analyze-image` → vision tool runs first (always, never skipped for medical safety) → visual description combined with user question → enriched query routed through RAG → grounded response returned.

**Voice output**
User clicks speaker on a bot message → POST to `/speak` → gTTS generates MP3 → streamed back → played at 1.25x speed → temp file auto-deleted.

### LangGraph Agent — 7 Nodes

1. **check_urgency** — runs first on every message; classifies severity using an LLM
2. **process_image** — runs if and only if an image is attached; never skipped
3. **decide_next_step** — routes to one of the next three nodes based on intent
4. **direct_chat** — handles greetings and small talk with conversation context
5. **clarify** — generates a single targeted follow-up question for vague input
6. **retrieve** — runs the RAG chain against the Gale Encyclopedia
7. **assemble_response** — prepends emergency warnings if urgency was flagged

> Emergency messages bypass clarification — a chest-pain user is never asked to rephrase.

---

## How RAG Was Built

1. **Ingest** — Gale Encyclopedia PDF split into 500-character chunks (20-character overlap) via `RecursiveCharacterTextSplitter`
2. **Embed** — Each chunk encoded to a 384-dimensional vector using `all-MiniLM-L6-v2`
3. **Index** — Vectors upserted into a Pinecone serverless index (cosine similarity)
4. **Retrieve** — User query embedded with the same model; top 3 semantically similar chunks returned
5. **Generate** — Retrieved chunks + conversation history stuffed into system prompt; Llama 3.3 70B generates a grounded answer

---

## Project Structure

```
├── streamlit_app.py       # Streamlit UI; launches FastAPI as a background thread
├── backend.py             # FastAPI app — /get, /analyze-image, /transcribe, /speak, /reset-memory
├── src/
│   ├── agent.py           # LangGraph state machine and node implementations
│   ├── tools.py           # Agent tools: RAG retrieval, vision, clarification, urgency
│   ├── helper.py          # PDF loading, Whisper transcription, image analysis, TTS
│   └── prompt.py          # System prompt with format detection and medical guardrails
├── store_index.py         # One-time script: chunk → embed → upsert to Pinecone
├── Dockerfile             # Container definition for Hugging Face Spaces
└── .github/workflows/     # GitHub Actions — auto-deploys to Hugging Face on push
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

FastAPI starts automatically as a background thread — no separate process needed.

Add a `.env` file with:
```
GROQ_API_KEY=your_key
PINECONE_API_KEY=your_key
```

---

## Safety & Limitations

MediBot includes a medical urgency classifier that screens every message for red-flag symptoms and surfaces emergency guidance. However, this chatbot is for **educational and informational purposes only** — it is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

**Known limitations:**
- Server-side memory resets on page reload (not yet per-user session)
- Relies on Llama 4 Scout vision model, currently in Groq's preview tier

---

## License

Apache 2.0
