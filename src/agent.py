from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from src.tools import (
    gale_rag_tool,
    vision_tool,
    symptom_clarifier_tool,
    urgency_classifier_tool,
)
import src.tools as _tools_module  # accessed at call-time so _llm/_memory are already injected


# ── Agent State ───────────────────────────────────────────────────────────────
# Tracks everything the agent knows as it moves through nodes.

class AgentState(TypedDict):
    user_message: str
    image_path: Optional[str]           # set when an image is attached
    visual_description: Optional[str]   # set by process_image node
    urgency_result: Optional[dict]      # set by check_urgency node
    retrieved_context: Optional[str]    # reserved for future use
    final_answer: Optional[str]         # set by clarify or retrieve, finalized by assemble_response
    status_updates: List[str]           # live progress messages shown to the user


# ── Node functions ─────────────────────────────────────────────────────────────
# Each node receives the full state and returns a dict of only the fields it updates.
# LangGraph merges these partial updates into the running state automatically.

def check_urgency(state: AgentState) -> dict:
    """Node 1 — always the first node. Runs the urgency classifier on every message.
    Acts as a safety net so emergency red flags (chest pain, stroke, etc.) are never missed."""
    return {
        "status_updates": state["status_updates"] + ["🔍 Checking urgency..."],
        "urgency_result": urgency_classifier_tool(state["user_message"]),
    }


def process_image(state: AgentState) -> dict:
    """Node 2 — runs ONLY when image_path is in state. Calls the Llama 4 Scout vision model
    for a clinical description BEFORE any retrieval. ALWAYS runs when an image is attached —
    this is non-negotiable for medical safety (never ignore visual evidence)."""
    return {
        "status_updates": state["status_updates"] + ["🖼️ Analyzing image..."],
        "visual_description": vision_tool(state["image_path"], state.get("user_message", "")),
    }


def decide_next_step(state: AgentState) -> dict:
    """Node 3 — explicit decision checkpoint in the graph. Exists as a visible node for
    interpretability. The actual routing logic lives in the conditional edge function
    route_from_decide() below — this node itself is a no-op state pass-through."""
    return {
        "status_updates": state["status_updates"] + ["🤔 Deciding next step..."],
    }


def direct_chat(state: AgentState) -> dict:
    """Node 4 — reached when input is chitchat, greeting, or non-medical small talk.
    Calls the LLM directly — no RAG, no Pinecone, no clarifier.
    Passes the last 2 real memory exchanges so the LLM references actual prior context
    instead of hallucinating. Saves the exchange to memory so greetings are part of history."""
    all_history = (
        _tools_module._memory.chat_memory.messages
        if _tools_module._memory else []
    )
    # Build readable history from the last 2 exchanges (up to 4 messages: 2 human + 2 AI)
    recent = all_history[-4:]
    history_text = ""
    for msg in recent:
        role = "Human" if getattr(msg, "type", "") == "human" else "MediBot"
        history_text += f"{role}: {msg.content}\n"

    prompt = (
        "You are MediBot, a friendly medical assistant powered by the Gale Encyclopedia of Medicine.\n"
        "The user has sent a greeting or non-medical message. Respond warmly and concisely.\n"
        "If it is a greeting with NO prior conversation: introduce yourself as MediBot from the "
        "Gale Encyclopedia of Medicine and ask 'How can I help you today?'.\n"
        "If it is a greeting WITH prior conversation shown below: welcome them back briefly and "
        "reference the actual topic discussed — do NOT invent or guess topics not shown.\n"
        "If it is a 'thank you' or acknowledgement: respond warmly and invite further questions.\n"
        "If it is a meta question ('who are you', 'what can you do'): briefly describe your purpose.\n"
        + (f"\nRecent conversation:\n{history_text}" if history_text else "")
        + f"\nUser message: \"{state['user_message']}\"\n\n"
        "Respond naturally. Keep it short — 1-3 sentences."
    )

    response = _tools_module._llm.invoke(prompt)
    answer = response.content.strip()

    # Save greeting/chitchat exchanges to memory so they're part of conversation history
    if _tools_module._memory:
        _tools_module._memory.save_context(
            {"input": state["user_message"]},
            {"output": answer}
        )

    return {
        "status_updates": state["status_updates"] + ["💬 Responding..."],
        "final_answer": answer,
    }


def clarify(state: AgentState) -> dict:
    """Node 4 — reached when input is vague AND non-emergency AND no image.
    Generates one targeted clarifying question instead of guessing and retrieving wrong content.
    A doctor asks questions before diagnosing — MediBot does too."""
    return {
        "status_updates": state["status_updates"] + ["💬 Asking a clarifying question..."],
        "final_answer": symptom_clarifier_tool(state["user_message"]),
    }


def retrieve(state: AgentState) -> dict:
    """Node 5 — builds the query (enriched with visual description when an image was processed),
    searches the Gale Encyclopedia via Pinecone, and generates a grounded answer with Llama 3.3 70B.
    This is where the describe-then-retrieve pattern for images executes."""
    vis = state.get("visual_description")
    user_q = state.get("user_message", "")

    if vis:
        # Image path: build the enriched query that steers Pinecone toward matching conditions
        if user_q:
            query = (
                f"A user has uploaded a medical image and is asking a question about it.\n\n"
                f"Visual observations from the image: {vis}\n\n"
                f"User's question: {user_q}\n\n"
                f"Based on the visual observations above, search the medical knowledge base for conditions "
                f"whose typical presentation matches this description — pay close attention to the shape, "
                f"pattern, color distribution, and texture described. Then answer the user's question."
            )
        else:
            query = (
                f"A user has uploaded a medical image.\n\n"
                f"Visual observations from the image: {vis}\n\n"
                f"Based on the visual observations above, search the medical knowledge base for conditions "
                f"whose typical presentation matches this description. Provide relevant medical information."
            )
    else:
        query = user_q

    # Always save the original user message to memory, not the enriched vision query
    answer = gale_rag_tool(query, memory_key=user_q if user_q else query)
    return {
        "status_updates": state["status_updates"] + [
            "📚 Searching medical knowledge base...",
            "💬 Generating answer...",
        ],
        "final_answer": answer,
    }


def assemble_response(state: AgentState) -> dict:
    """Node 6 — final node after retrieve. Prepends the urgency warning to the answer
    if the urgency classifier flagged this as moderate or emergency severity."""
    urgency = state.get("urgency_result") or {}
    final = state.get("final_answer", "")
    if urgency.get("is_urgent") and urgency.get("warning_message"):
        final = f"{urgency['warning_message']}\n\n---\n\n{final}"
    return {"final_answer": final}


# ── Conditional edge routing functions ────────────────────────────────────────

def route_after_urgency(state: AgentState) -> str:
    """After check_urgency: if an image is attached, process it first.
    Otherwise jump straight to the decision node."""
    return "process_image" if state.get("image_path") else "decide_next_step"


def route_from_decide(state: AgentState) -> str:
    """After decide_next_step: apply routing rules in priority order.

    Rule 1 (safety override): emergency severity ALWAYS goes to retrieve — never ask
    a potential cardiac patient to rephrase their question. (Risk 8 from analysis.)
    Rule 2: if an image was analyzed, always retrieve (describe-then-retrieve pattern).
    Rule 3: LLM classifier picks chitchat | vague_medical | clear_medical | emergency.
      - chitchat  → direct_chat (friendly reply, no RAG)
      - vague_medical → clarify
      - clear_medical / emergency → retrieve
    Safe fallback on any classifier failure: "clear_medical" → retrieve.
    """
    urgency = state.get("urgency_result") or {}
    if urgency.get("severity") == "emergency":
        return "retrieve"
    if state.get("visual_description"):
        return "retrieve"

    # LLM-based intent classifier with conversation context to catch follow-up questions
    user_msg = state.get("user_message", "")
    try:
        # Pass last 2 memory messages so the classifier can detect follow-ups in context
        all_history = (
            _tools_module._memory.chat_memory.messages
            if _tools_module._memory else []
        )
        recent_ctx = ""
        if all_history:
            for msg in all_history[-2:]:
                role = "Human" if getattr(msg, "type", "") == "human" else "MediBot"
                recent_ctx += f"{role}: {msg.content}\n"

        classification_prompt = (
            "Classify the following user message into exactly one of these four categories.\n\n"
            "Categories:\n"
            "  chitchat      — ONLY pure greetings ('hello', 'hi', 'hey'), thanks, farewell, "
            "or meta questions about the bot itself ('who are you', 'what can you do'). "
            "Never classify a message as chitchat if it relates to health or follows a medical discussion.\n"
            "  vague_medical — a medical topic is mentioned but too vague to look up "
            "('I have a rash', 'I don't feel good', 'I have dots', 'something hurts')\n"
            "  clear_medical — specific medical question OR a follow-up to a prior medical discussion. "
            "Examples: 'what is diabetes', 'symptoms of hypertension', 'in short', 'what should I do', "
            "'tell me more', 'summarize', 'any advice', 'what next', 'yes', 'no', 'it does', 'so?'.\n"
            "  emergency     — urgent medical red flag "
            "('chest pain', 'can\\'t breathe', 'stroke symptoms', 'severe bleeding')\n\n"
            "IMPORTANT RULE: If there is recent medical conversation context below AND the message "
            "could be a follow-up (asking for next steps, a summary, saying yes/no, or asking "
            "'in short / tell me more / what should I do / any advice'), classify as clear_medical, "
            "NOT chitchat.\n\n"
            + (f"Recent conversation context:\n{recent_ctx}\n" if recent_ctx else "")
            + f"Message to classify: \"{user_msg}\"\n\n"
            "Respond with ONLY the label — one word, no punctuation: "
            "chitchat, vague_medical, clear_medical, or emergency"
        )
        response = _tools_module._llm.invoke(classification_prompt)
        label = response.content.strip().lower().strip('"').strip("'")
        if label == "chitchat":
            return "direct_chat"
        if label == "vague_medical":
            return "clarify"
        # emergency or clear_medical both go to retrieve
        if label in ("emergency", "clear_medical"):
            return "retrieve"
    except Exception as e:
        print(f"Input classifier error (safe fallback to retrieve): {e}")

    return "retrieve"  # safest fallback — RAG on anything unclassified


# ── Graph construction ─────────────────────────────────────────────────────────

def _build_agent():
    workflow = StateGraph(AgentState)

    # Register all seven nodes (direct_chat added for chitchat routing)
    workflow.add_node("check_urgency", check_urgency)
    workflow.add_node("process_image", process_image)
    workflow.add_node("decide_next_step", decide_next_step)
    workflow.add_node("direct_chat", direct_chat)
    workflow.add_node("clarify", clarify)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("assemble_response", assemble_response)

    # Entry point: always start with urgency check
    workflow.set_entry_point("check_urgency")

    # After urgency: branch on whether an image is attached
    workflow.add_conditional_edges(
        "check_urgency",
        route_after_urgency,
        {"process_image": "process_image", "decide_next_step": "decide_next_step"},
    )

    # After image processing: always go to the decision node
    workflow.add_edge("process_image", "decide_next_step")

    # After decision node: branch on intent (chitchat / vague_medical / clear_medical / emergency)
    workflow.add_conditional_edges(
        "decide_next_step",
        route_from_decide,
        {"direct_chat": "direct_chat", "clarify": "clarify", "retrieve": "retrieve"},
    )

    # direct_chat ends the graph — no RAG, no urgency prepend needed
    workflow.add_edge("direct_chat", END)

    # Clarify ends the graph — the emergency override ensures no true emergency ever reaches this
    workflow.add_edge("clarify", END)

    # Retrieve feeds into assembly, then done
    workflow.add_edge("retrieve", "assemble_response")
    workflow.add_edge("assemble_response", END)

    return workflow.compile()


# Compiled agent — built at import time, tools are injected before first call via init_tools()
_agent = _build_agent()


def run_agent(user_message: str, image_path: str = None) -> dict:
    """Public entry point called by app.py routes. Builds the initial state and
    invokes the compiled LangGraph agent. Returns the final state dict."""
    initial_state: AgentState = {
        "user_message": user_message or "",
        "image_path": image_path,
        "visual_description": None,
        "urgency_result": None,
        "retrieved_context": None,
        "final_answer": None,
        "status_updates": [],
    }
    return _agent.invoke(initial_state)
