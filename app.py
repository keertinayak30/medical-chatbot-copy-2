from flask import Flask, render_template, jsonify, request, send_file, after_this_request
import tempfile
import os
from src.helper import download_hugging_face_embeddings, transcribe_audio, analyze_image, text_to_speech
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from src.tools import init_tools
from src.agent import run_agent

app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


index_name = "medchatbot2" 



embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    groq_api_key=GROQ_API_KEY,
    temperature=0.3,
    streaming=True
)

memory = ConversationBufferWindowMemory(
    k=5, 
    memory_key="chat_history",
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Inject shared chain objects into the tools module — must happen before any route is called
init_tools(rag_chain, llm, memory)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={
        "prompt": ChatPromptTemplate.from_messages([
            ("system", system_prompt),MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{context}\n\nQuestion: {question}")
        ])
    },
    return_source_documents=True,
    verbose=False 
)


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["POST"])
def chat():
    if request.is_json:
        data = request.get_json()
        msg = data.get('msg', '')
    else:
        msg = request.form.get('msg', '')

    if not msg:
        return jsonify({"response": "Please enter a message!"}), 400

    print(f"User: {msg}")

    # Agent handles urgency check, vagueness detection, RAG retrieval, and response assembly
    result = run_agent(user_message=msg, image_path=None)
    print(f"Bot: {result['final_answer'][:120]}...")

    return jsonify({"response": result["final_answer"], "status_updates": result["status_updates"]})


@app.route("/reset-memory", methods=["POST"])
def reset_memory():
    memory.chat_memory.clear()
    return jsonify({"status": "ok"})


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    if not data or not data.get('text', '').strip():
        return jsonify({"error": "No text provided"}), 400

    text = data['text'].strip()
    tmp_path = text_to_speech(text)

    if tmp_path is None:
        return jsonify({"error": "TTS generation failed. Check server logs."}), 500

    # Delete the temp MP3 after Flask finishes streaming it — no files pile up on disk
    @after_this_request
    def cleanup(response):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return response

    return send_file(tmp_path, mimetype='audio/mpeg')


@app.route("/analyze-image", methods=["POST"])
def analyze_image_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    user_question = request.form.get('question', '').strip()

    # Validate file extension — only jpg, jpeg, png, webp accepted
    filename = image_file.filename or ''
    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    if ext not in allowed_exts:
        return jsonify({"error": f"Unsupported format '{ext}'. Please upload a JPG, PNG, or WebP image."}), 400

    # Validate file size before saving — Groq vision API enforces a 20 MB limit
    image_file.seek(0, 2)
    file_size = image_file.tell()
    image_file.seek(0)
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB
    if file_size > MAX_IMAGE_SIZE:
        return jsonify({"error": "Image too large. Maximum size is 20 MB (Groq vision API limit)."}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            image_file.save(tmp_path)

        # Agent handles vision description, enriched retrieval, urgency check, and response assembly
        result = run_agent(user_message=user_question, image_path=tmp_path)

        if result.get("final_answer") is None:
            return jsonify({"error": "Agent could not generate a response"}), 500

        print(f"Bot (image): {result['final_answer'][:120]}...")
        return jsonify({"response": result["final_answer"], "status_updates": result["status_updates"]})

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']

    # Save the uploaded audio to a temp file, then transcribe and clean up
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)

        text = transcribe_audio(tmp_path)
        return jsonify({"text": text})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host="0.0.0.0", port=port, debug=True)