# This is where you actually set your "SDK version" (Python version)
FROM python:3.10-slim

WORKDIR /app

# Standard Hugging Face user setup to avoid permission errors
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN pip install --no-cache-dir -r requirements.txt

# Streamlit serves the UI on 7860 (the only port HF Spaces exposes).
# FastAPI starts as a background daemon thread inside the Streamlit process on 127.0.0.1:8000.
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]