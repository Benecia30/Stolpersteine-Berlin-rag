FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# CPU-only PyTorch
RUN uv pip install --system \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Runtime dependencies
RUN uv pip install --system \
    streamlit \
    groq \
    python-dotenv \
    pandas \
    numpy \
    scipy \
    rank-bm25 \
    faiss-cpu \
    sentence-transformers

COPY app.py ./
COPY scripts ./scripts
COPY utils ./utils
COPY pages ./pages
COPY data ./data

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
