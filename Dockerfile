FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

COPY models/ ./models/
COPY data/ ./data/

# Copy application code
COPY main.py .
COPY myscript/ ./myscript/
COPY config/ ./config/

# Create necessary directories with proper permissions
RUN mkdir -p /app/output /app/temp && \
    chmod 777 /app/output /app/temp

# Install PyTorch CPU-only version
# Thêm "numpy<2.0" vào danh sách cài đặt
RUN pip install --no-cache-dir \
    "numpy==1.23.5" \
    torch==2.4.1+cpu \
    torchvision==0.19.1+cpu \
    torchaudio==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy EasyMocap source code
COPY setup.py .
COPY easymocap/ ./easymocap/
COPY myeasymocap/ ./myeasymocap/
COPY apps/ ./apps/
COPY config/ ./config/

# Install EasyMocap in development mode
RUN python setup.py develop

# Copy application code
COPY main.py .
COPY myscript/ ./myscript/

# Create necessary directories
RUN mkdir -p /app/output /app/temp

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Start command
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:5000", "--timeout", "900", "--workers", "1", "--worker-class", "sync", "--worker-tmp-dir", "/dev/shm"]
