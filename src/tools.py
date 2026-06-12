import os
import json
from src.helper import analyze_image

# Module-level references — all None until init_tools() is called from app.py at startup.
# This pattern avoids circular imports: app.py creates the chain objects, then injects them here.
_rag_chain = None
_llm = None
_memory = None


def init_tools(rag_chain, llm, memory):
    """Called once from app.py after chain setup. Injects shared objects so tools can
    use rag_chain, llm, and memory without creating circular imports."""
    global _rag_chain, _llm, _memory
    _rag_chain = rag_chain
    _llm = llm
    _memory = memory


# ── Tool 1: Gale Encyclopedia RAG retrieval + Llama 3.3 70B generation ──────

def gale_rag_tool(query: str, memory_key: str = None) -> str:
    """Run a query through Pinecone retrieval and the Llama 3.3 70B chain.
    Appends recent memory context, returns the grounded answer, and saves to memory.
    memory_key: the original user message to store in memory (defaults to query).
    Pass the original user text when query is an enriched vision prompt so memory
    stays readable rather than containing internal prompt plumbing."""
    recent_history = [
        h.content for h in _memory.chat_memory.messages[-3:] if hasattr(h, 'content')
    ]
    full_query = (
        query if not recent_history
        else f"{query} (previous: {' '.join(recent_history[-1:])})"
    )
    result = _rag_chain.invoke({"input": full_query, "chat_history": _memory.chat_memory.messages})
    answer = result["answer"]
    _memory.save_context({"input": memory_key or query}, {"output": answer})
    return answer


# ── Tool 2: Vision analysis — Groq Llama 4 Scout ────────────────────────────

def vision_tool(image_path: str, user_question: str = "") -> str:
    """Base64-encode the image and return a clinical visual description from the vision model.
    Wraps analyze_image() from helper.py — reuses it, does not rebuild it."""
    return analyze_image(image_path, user_question)


# ── Tool 3: Symptom clarifier — targeted follow-up for vague input ───────────

def symptom_clarifier_tool(user_message: str) -> str:
    """When input is too vague to retrieve meaningful content, generate one specific,
    empathetic clarifying question instead of guessing and pulling the wrong Gale pages."""
    prompt = (
        "You are MediBot, a medical assistant. The user's message is too vague to answer accurately.\n"
        "Generate ONE short, specific, empathetic follow-up question to gather the information needed.\n"
        "Ask about: location on body, duration, severity, associated symptoms, or relevant history.\n"
        f'User message: "{user_message}"\n'
        "Output only the clarifying question — no preamble, no explanation."
    )
    response = _llm.invoke(prompt)
    return response.content.strip()


# ── Tool 4: Urgency classifier — red-flag detector, runs on every message ────

def urgency_classifier_tool(user_message: str) -> dict:
    """Classify the message for medical urgency. Detects red flags: chest pain,
    difficulty breathing, stroke symptoms (FAST), severe bleeding, suicidal ideation,
    anaphylaxis, loss of consciousness. Returns is_urgent, severity, and warning_message.
    Falls back to a safe non-urgent result if the LLM returns malformed JSON."""
    prompt = (
        "You are a medical triage assistant. Analyze the following message for medical urgency.\n"
        "Red flags to detect: chest pain, difficulty breathing, suicidal ideation or self-harm, "
        "severe bleeding, stroke symptoms (face drooping, arm weakness, speech difficulty, sudden "
        "severe headache), anaphylaxis signs, loss of consciousness, signs of poisoning or overdose.\n\n"
        f'Message: "{user_message}"\n\n'
        "Respond ONLY with a valid JSON object, no markdown, no extra text:\n"
        '{"is_urgent": true|false, "severity": "none"|"moderate"|"emergency", '
        '"warning_message": "string or null"}\n\n'
        "For emergency severity, warning_message MUST tell the user to call 112 immediately "
        "or go to the nearest hospital. The user is in Mumbai, India. Start with ⚠️."
    )
    try:
        response = _llm.invoke(prompt)
        content = response.content.strip()
        # Strip markdown code fences if the LLM wraps the JSON in ```json ... ```
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else content
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception as e:
        print(f"Urgency classifier error (safe fallback applied): {e}")
        return {"is_urgent": False, "severity": "none", "warning_message": None}
