# PilotView
# Using official uv image with Python 3.12

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install FFmpeg for video transcoding
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source
COPY main.py ./
COPY templates ./templates/

# Install the project
RUN uv sync --frozen --no-dev

# Environment variables
ENV COMMA_DATA_DIR=/data \
    COMMA_CACHE_DIR=/cache

# Create directories for data and cache
RUN mkdir -p /data /cache

EXPOSE 5000

CMD ["uv", "run", "python", "main.py", "--host", "0.0.0.0", "--port", "5000"]
