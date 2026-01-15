FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # ===> THÊM MỚI: Ép PyTorch/NumPy chạy đơn luồng để tránh treo máy t3.micro <===
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONPATH=/app

# 1. Install system dependencies
# Gom nhóm lại để giảm số layer và dung lượng image
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# 2. Install PyTorch CPU-only version (Cài trước để tận dụng cache)
# Sử dụng đúng phiên bản bạn đã chọn
RUN pip install --no-cache-dir \
    "numpy==1.23.5" \
    torch==2.4.1+cpu \
    torchvision==0.19.1+cpu \
    torchaudio==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Copy requirements (nếu có thêm thư viện khác ngoài torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy toàn bộ mã nguồn
COPY . .

# 5. Install EasyMocap
# Lệnh này giúp python nhận diện package 'apps' và 'myeasymocap'
RUN pip install .

# 6. Create necessary directories
RUN mkdir -p /app/output /app/temp /app/models && \
    chmod 777 /app/output /app/temp /app/models

# Expose port
EXPOSE 5000

# Health check (Tăng thời gian start-period vì load AI lần đầu rất lâu)
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# Start command
# --timeout 3600: Cho phép upload/xử lý lâu mà không bị kill kết nối
# --workers 1: BẮT BUỘC cho logic Single Job
# --threads 4: Cho phép xử lý request status trong khi đang upload file
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "3600", "--worker-class", "gthread", "--worker-tmp-dir", "/dev/shm"]