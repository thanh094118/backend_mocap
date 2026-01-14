#!/bin/bash

# 1. Tải Model trước
echo "--- [STARTUP] Checking and Downloading Models... ---"
python download_models.py

# Kiểm tra nếu tải thất bại thì dừng container luôn (để Render biết là lỗi)
if [ $? -ne 0 ]; then
    echo "--- [ERROR] Model download failed. Exiting. ---"
    exit 1
fi

echo "--- [STARTUP] Models ready. Starting Gunicorn Server... ---"

# 2. Khởi động Gunicorn (Lệnh cũ trong CMD của bạn chuyển vào đây)
# Dùng 'exec' để gunicorn nhận được các tín hiệu hệ thống (như SIGTERM)
exec gunicorn main:app \
    --bind 0.0.0.0:5000 \
    --timeout 900 \
    --workers 1 \
    --worker-class sync \
    --worker-tmp-dir /dev/shm