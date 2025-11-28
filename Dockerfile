FROM python:3.11-slim

# --- Basic runtime settings ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ffmpeg \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv using pip ---
RUN pip install --no-cache-dir uv

# --- App setup ---
WORKDIR /app

# Copy dependency files first for cache efficiency
COPY pyproject.toml uv.lock ./

# Install deps based on the lockfile
RUN uv sync --frozen --no-dev

# Install playwright (required for scraping)
RUN uv run playwright install --with-deps chromium

# Copy full source after dependencies are installed
COPY . .

# Ensure your runtime workspace exists
RUN mkdir -p /app/LLMFiles

# Expose FastAPI port
EXPOSE 7860

# Start API
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
