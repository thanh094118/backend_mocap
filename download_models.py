# ============================================
# FILE 2: download_models.py (MINIMAL)
# ============================================
import gdown
import sys
import os
from pathlib import Path

# Tự động nhận diện thư mục chứa file này (thường là /app trong Docker)
BASE_DIR = Path(__file__).parent.absolute()

MODELS = {
    "pose_hrnet": {
        "id": "1eZPkFzRN_TL_tUvfRTofiqWngproIirZ",
        "output": "models/pose_hrnet_w48_384x288.pth"
    },
    "pare_checkpoint": {
        "id": "1SRrH_ha122KD4z_PjJ0UDm_4U1ti97X-",
        "output": "models/pare_w_3dpw_checkpoint.ckpt"
    }
}

def main():
    print(f"Running download script from: {BASE_DIR}")

    for name, config in MODELS.items():
        # Tạo đường dẫn tuyệt đối: /app/models/...
        output_path = (BASE_DIR / config["output"]).resolve()
        
        # 1. Kiểm tra nếu file đã có và không bị rỗng
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[SKIP] {name} exists ({output_path.stat().st_size} bytes)")
            continue

        # 2. Tạo thư mục cha nếu chưa có (mkdir -p)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Tải file (Fail Fast: Lỗi là dừng build ngay)
        print(f"[DOWNLOADING] {name}...")
        try:
            url = f"https://drive.google.com/uc?id={config['id']}"
            # quiet=False để hiện thanh progress bar của gdown (hữu ích khi chờ đợi)
            gdown.download(url, str(output_path), quiet=False, fuzzy=True)

            # Check lại lần cuối
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise Exception("File downloaded but is empty/missing")
                
        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")
            # Xóa file lỗi để tránh cache hỏng
            if output_path.exists(): os.remove(output_path)
            sys.exit(1) # Exit 1 để Docker biết là Build Fail

    print("[DONE] All models ready.")

if __name__ == "__main__":
    main()