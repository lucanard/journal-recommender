FROM python:3.11-slim

WORKDIR /app

# Install dependencies (no sentence-transformers — use API embeddings)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY app.py recommender.py vector_store.py report_generator.py ./
COPY 05_generate_embeddings.py ./

# Copy data
COPY data/ ./data/

# Copy frontend
COPY index.html ./

EXPOSE 8000

# Start with configurable LLM and embedding provider via env vars
CMD ["python", "app.py", "--data-dir", "data", "--host", "0.0.0.0", "--port", "8000"]
