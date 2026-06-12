from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
import os
import base64
import tempfile
from groq import Groq
from gtts import gTTS

# Vision model used for image analysis — Scout is currently Groq Preview tier.
# If the model is deprecated, swap this one constant and nothing else needs to change.
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings


# Transcribe audio using Groq Whisper API (whisper-large-v3)
# Takes a local audio file path, returns the transcribed text string
def transcribe_audio(audio_file_path):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="text"
        )
    return transcription


# Analyze an image using Groq's vision model (VISION_MODEL).
# Reads the image, base64-encodes it, and asks the model for a clinical visual description.
# user_question is passed as context so the description targets what the user cares about.
# Returns the description string. Caller is responsible for try/except.
def analyze_image(image_file_path, user_question):
    # Map file extension to MIME type for the base64 data URL
    ext = os.path.splitext(image_file_path)[1].lower().lstrip('.')
    mime_map = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}
    mime_type = mime_map.get(ext, 'image/jpeg')

    with open(image_file_path, 'rb') as f:
        b64_image = base64.b64encode(f.read()).decode('utf-8')

    # Add user question as context so the description is more targeted
    context_hint = f" The user is asking about: {user_question}." if user_question else ""
    vision_prompt = (
        "You are a medical visual observer. Describe this image using specific descriptive vocabulary "
        "that would appear in a medical encyclopedia. Include all of the following when visible:\n"
        "- Shape: use precise terms like circular, oval, annular, ring-shaped, linear, irregular, "
        "well-defined, poorly-defined\n"
        "- Color and color distribution: uniform, central clearing, raised border, erythematous, "
        "hyperpigmented, hypopigmented\n"
        "- Texture: scaly, smooth, raised, flat, crusted, vesicular, pustular, weeping\n"
        "- Size: approximate dimensions if estimable\n"
        "- Location on body: if visible\n"
        "- Pattern: clustered, isolated, symmetrical, spreading outward\n"
        "Use common dermatological descriptive vocabulary. Do NOT diagnose or name conditions. "
        f"Only describe what you observe in concrete visual terms. Be specific about shapes, borders, "
        f"and textures.{context_hint}"
    )

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
                    },
                    {
                        "type": "text",
                        "text": vision_prompt
                    }
                ]
            }
        ],
        max_tokens=512
    )
    return response.choices[0].message.content


# Convert text to speech using gTTS (Google Text-to-Speech — free, no API key needed).
# Saves the audio to a temp MP3 file and returns the path.
# Returns None and logs the error if gTTS fails — caller handles the None case gracefully.
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            tmp_path = tmp.name
        tts.save(tmp_path)
        return tmp_path
    except Exception as e:
        print(f"TTS error: {e}")
        return None