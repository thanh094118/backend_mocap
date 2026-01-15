FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 1. Install system dependencies (Đã thêm unzip)
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    unzip \
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
ENV PYTHONPATH=/app
# 2. Install PyTorch CPU-only version (Cài trước để tận dụng cache)
RUN pip install --no-cache-dir \
    "numpy==1.23.5" \
    torch==2.4.1+cpu \
    torchvision==0.19.1+cpu \
    torchaudio==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy mã nguồn (Gộp các lệnh COPY lại cho gọn)
COPY . .

# 6. Install EasyMocap (FIX QUAN TRỌNG: dùng pip install thay vì setup.py develop)
# Điều này giúp importlib tìm thấy metadata của gói easymocap
RUN pip install .

# 7. Tải Model (Chạy sau khi đã COPY code vào)
# Lưu ý: Nếu bạn muốn tải model lúc build image thì bỏ comment dòng dưới.
# Tuy nhiên, nếu model quá nặng, nên để nó chạy trong start.sh hoặc tải tay bên ngoài.
# RUN python download_models.py

# 8. Create necessary directories
RUN mkdir -p /app/output /app/temp /app/models && \
    chmod 777 /app/output /app/temp /app/models

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Start command
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:5000", "--timeout", "900", "--workers", "1", "--worker-class", "sync", "--worker-tmp-dir", "/dev/shm"]
