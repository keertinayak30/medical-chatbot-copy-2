---
title: Medical Chatbot V2
emoji: 👩🏻‍⚕️
colorFrom: green
colorTo: blue
sdk: docker
python_version: 3.10
app_file: streamlit_app.py
pinned: false
license: apache-2.0
---

# MediBot AI: Multimodal Agentic Medical Assistant

**Live Demo:** [Launch MediBot V2](https://keertinayak30-medical-chatbot-v2.hf.space/)

## Overview

MediBot is an end-to-end multimodal medical Q&A chatbot grounded in the Gale Encyclopedia of Medicine. Users can interact via text, voice, or by uploading an image of a visible symptom (such as a skin condition), and the system returns a grounded answer retrieved from a vector database of medical literature. On top of the core RAG pipeline, MediBot uses an agentic LangGraph layer that classifies intent, asks clarifying questions when input is vague, and triggers safety warnings for medical emergencies.

## Features

- **Retrieval-Augmented Generation (RAG):** Answers are grounded in the Gale Encyclopedia of Medicine, indexed in Pinecone, to minimize hallucinations.
- **Voice Input:** Users can speak their symptoms; audio is transcribed via OpenAI Whisper hosted on Groq.
- **Image Analysis:** Users can upload photos of visible symptoms (e.g., rashes); a vision model produces a clinical description that is fed back into the RAG pipeline using a describe-then-retrieve pattern.
- **Voice Output:** Bot responses can be played aloud on demand using Google Text-to-Speech, with markdown stripped for natural speech.
- **Agentic Reasoning (LangGraph):** A 7-node state machine routes each message based on intent — direct chat, clarifying question, knowledge retrieval, or emergency response.
- **Urgency Detection:** Every message is screened for medical red flags (chest pain, breathing difficulty, suicidal ideation, severe bleeding, stroke symptoms). Emergencies trigger an immediate warning prepended to the response.
- **Symptom Clarification:** When input is too vague to answer accurately, the agent generates a focused follow-up question instead of hallucinating.
- **Conversational Memory:** Per-session memory window of the last 5 exchanges enables natural follow-up questions.
- **Markdown Rendering:** Bot responses support rich formatting including bold, lists, and tables, rendered live in the chat.
- **Adjustable Playback Speed:** Voice responses play back at 1.25x for faster listening.

## Tech Stack

- **Backend API:** FastAPI with uvicorn — exposes agent and helpers as JSON endpoints (/get, /analyze-image, /transcribe, /speak, /reset-memory)
- **LLM Orchestration:** LangChain
- **Agent Framework:** LangGraph (state machine with conditional routing)
- **Large Language Model:** Llama 3.3 70B Versatile via Groq Cloud
- **Vision Model:** Llama 4 Scout 17B via Groq Cloud
- **Speech-to-Text:** Whisper Large V3 via Groq Cloud
- **Text-to-Speech:** Google Text-to-Speech (gTTS)
- **Vector Database:** Pinecone (serverless, cosine similarity, 384 dimensions)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Frontend:** Streamlit — st.chat_message, st.chat_input, st.file_uploader, st.audio, audio-recorder-streamlit for mic input; communicates with FastAPI via httpx
- **Deployment:** Docker on HuggingFace Spaces with GitHub Actions CI/CD; FastAPI runs as a background daemon thread (127.0.0.1:8000), Streamlit serves the UI on port 7860

## Architecture

### Request Flow

**Text query:** User input → Flask `/get` route → LangGraph agent → urgency check → intent classification → routed to direct chat, clarification, or RAG retrieval → Pinecone vector search → Llama 3.3 70B response → assembled with safety warnings if applicable → returned to frontend.

**Voice query:** Microphone capture → MediaRecorder API → POST to `/transcribe` → Whisper transcription → text inserted into input box → user reviews and sends → same flow as text query.

**Image query:** Image upload → POST to `/analyze-image` → image validated and saved temporarily → agent runs vision tool first (always, never skipped for medical safety) → visual description combined with user question into an enriched query → routed through RAG retrieval → grounded response returned.

**Voice output:** User clicks speaker button on a bot message → markdown stripped via invisible DOM element → POST to `/speak` → gTTS generates MP3 → streamed back via Flask `send_file` → played in browser at 1.25x speed → temp file auto-deleted.

### Agent State Graph

The LangGraph agent contains seven nodes:

1. **check_urgency** — runs first on every message, classifies severity using an LLM
2. **process_image** — runs if and only if an image is attached; never skipped
3. **decide_next_step** — routes to one of the next three nodes based on intent classification
4. **direct_chat** — handles greetings and small talk with conversation context
5. **clarify** — generates a single targeted follow-up question for vague medical input
6. **retrieve** — runs the RAG chain against the Gale Encyclopedia
7. **assemble_response** — prepends emergency warnings if the urgency check flagged the message

Routing priority ensures that emergency messages bypass clarification (you never ask a chest-pain user to rephrase) and that uploaded images always go through visual analysis before any retrieval decision.

## How RAG Was Built

1. **Data ingestion:** The Gale Encyclopedia of Medicine PDF is split into 500-character chunks with 20-character overlap using LangChain's `RecursiveCharacterTextSplitter`.
2. **Embedding:** Each chunk is converted to a 384-dimensional vector using HuggingFace's `all-MiniLM-L6-v2` sentence transformer.
3. **Indexing:** Vectors are upserted into a Pinecone serverless index with cosine similarity.
4. **Retrieval:** At query time, the user question (or vision-enriched query) is embedded with the same model, and the top 3 most semantically similar chunks are returned.
5. **Generation:** Retrieved chunks are stuffed into a system prompt along with the user question and conversation history, and Llama 3.3 70B generates a grounded answer.

## Project Structure

- `app.py` — Flask routes, RAG chain setup, agent initialization
- `src/agent.py` — LangGraph state machine and node implementations
- `src/tools.py` — Four agent tools: RAG retrieval, vision analysis, symptom clarification, urgency classification
- `src/helper.py` — Utility functions for PDF loading, Whisper transcription, image analysis, and text-to-speech
- `src/prompt.py` — System prompt for the main LLM with format detection and medical guardrails
- `store_index.py` — One-time script to chunk, embed, and upload the Gale Encyclopedia to Pinecone
- `templates/chat.html` — Chat interface with input controls for text, voice, and image
- `static/` — CSS and JavaScript assets for the frontend
- `Dockerfile` — Container definition for HuggingFace Spaces deployment
- `.github/workflows/main.yml` — GitHub Actions workflow for auto-deploying to HuggingFace Spaces

## Safety and Limitations

MediBot includes a medical urgency classifier that screens every message for red-flag symptoms and surfaces emergency guidance for the user's region (Mumbai 112). However, this chatbot is intended for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Users should always consult a qualified healthcare professional for medical concerns.

Known limitations include: a global server-side memory object that resets on page reload (not yet per-user), client-side simulated streaming for status updates rather than true Server-Sent Events, and reliance on the Llama 4 Scout vision model which is currently in Groq's preview tier.

## License

Apache 2.0