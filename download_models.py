# ============================================
# FILE 2: download_models.py (FIXED)
# ============================================
"""
Download models from Google Drive - Smart Skip Logic
"""
import gdown
import os
import sys
from pathlib import Path

# Xác định thư mục gốc dựa trên vị trí file script này
# Giúp chạy đúng dù bạn đang đứng ở bất kỳ đâu trong terminal
BASE_DIR = Path(__file__).parent.absolute()

# Model configurations: (gdrive_id, output_path relative to BASE_DIR)
MODELS = {
    "pose_hrnet": {
        "id": "1eZPkFzRN_TL_tUvfRTofiqWngproIirZ",
        "output": "data/models/pose_hrnet_w48_384x288.pth"
    },
    "pare_checkpoint": {
        "id": "1SRrH_ha122KD4z_PjJ0UDm_4U1ti97X-",
        "output": "models/pare/data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt"
    },
    "smpl_neutral": {
        "id": "1Rza5kVxB7Lp5lP_o0r3LCpmk1LaepiI-",
        "output": "models/pare/data/body_models/smpl/SMPL_NEUTRAL.pkl"
    },
    "yolov5": {
        "id": "1bv56ZN7tRIoPXPfeow26rBdC1NddLdS2",
        "output": "models/yolo/yolov5m.pt"
    }
}

def download_file(file_id, output_path):
    """Download file from Google Drive"""
    # Sử dụng gdown với option fuzzy=True để tránh lỗi trích xuất ID
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Đảm bảo đường dẫn là tuyệt đối
    abs_output_path = (BASE_DIR / output_path).resolve()
    
    # Create directory if not exists
    output_dir = abs_output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading to {abs_output_path}...")
    
    try:
        # download với resume=True để nếu đứt mạng có thể nối lại (tùy server hỗ trợ)
        # output bắt buộc phải convert sang string cho gdown
        gdown.download(url, str(abs_output_path), quiet=False, fuzzy=True)
        
        # Kiểm tra lại lần cuối xem file có tải về thật không
        if abs_output_path.exists() and abs_output_path.stat().st_size > 0:
            print(f"✓ Downloaded: {output_path}")
            return True
        else:
            print(f"✗ Download finished but file is empty or missing.")
            return False
            
    except Exception as e:
        print(f"✗ Failed to download {output_path}: {e}")
        # Xóa file lỗi/rỗng nếu có để lần sau tải lại
        if abs_output_path.exists():
            os.remove(abs_output_path)
        return False

def main():
    """Download all models"""
    print("=" * 60)
    print("Downloading models (Smart Skip enabled)...")
    print(f"Base Directory: {BASE_DIR}")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for model_name, config in MODELS.items():
        print(f"\n[{model_name}]")
        
        # Xử lý đường dẫn
        target_path = BASE_DIR / config["output"]
        
        # === [LOGIC FIX QUAN TRỌNG] ===
        # Chỉ bỏ qua nếu file tồn tại VÀ dung lượng > 0 bytes
        if target_path.exists():
            if target_path.stat().st_size > 0:
                print(f"✓ Found valid file (Size: {target_path.stat().st_size / 1024 / 1024:.2f} MB)")
                print(f"  Skipping download: {config['output']}")
                success_count += 1
                continue
            else:
                print(f"⚠ File exists but is empty (0 bytes). Re-downloading...")
                os.remove(target_path) # Xóa file rỗng đi
        
        # Download
        if download_file(config["id"], config["output"]):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"Download Summary: {success_count} success, {fail_count} failed")
    print("=" * 60)
    
    if fail_count > 0:
        print("\n⚠ Some downloads failed. Please check the logs.")
        # Trả về exit code 1 để Docker biết là build fail
        sys.exit(1)
    else:
        print("\n✓ All models ready!")
        sys.exit(0)

if __name__ == "__main__":
    main()