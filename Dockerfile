# Hugging Face Spaces (Docker SDK) expects the app on port 7860 and runs the
# container as a non-root user with uid 1000.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/huggingface \
    PORT=7860 \
    OMP_NUM_THREADS=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . /app

USER user

# Bake the model into the image. Without this the first request after every
# cold start pays a ~90MB download before it does any work.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "75"]
