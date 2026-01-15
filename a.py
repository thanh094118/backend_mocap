import importlib.metadata
import sys
from packaging import version

# Danh sách các gói cần kiểm tra (Dựa trên file requirements.txt đã gộp)
REQUIRED_PACKAGES = {
    # --- PyTorch Stack ---
    "torch": "2.4.1",
    "torchvision": "0.19.1",
    "torchaudio": "2.4.1",
    
    # --- Core & Math ---
    "numpy": "1.23.5",
    "pandas": "1.5.3",
    "seaborn": "0.13.2",
    "joblib": "1.5.1",
    "tqdm": "4.67.1",
    "func-timeout": "4.3.5",
    
    # --- CV & Image ---
    "opencv-python": "4.5.0", # Lưu ý: check metadata name
    "scikit-image": "0.24.0",
    "lpips": "0.1.4",
    "ultralytics": "8.3.174",
    
    # --- 3D Rendering ---
    "pytorch3d": "0.7.8",
    "chumpy": "0.70",
    "pyrender": "0.1.45",
    "PyOpenGL": "3.1.0",
    "PyOpenGL-accelerate": None, # Chỉ cần có là được
    
    # --- DL Utils ---
    "pytorch-lightning": "2.5.5",
    "tensorboard": "2.8.0",
    "torch-tb-profiler": "0.4.3",
    "protobuf": "3.20.3",
    
    # --- Tools ---
    "gdown": "5.2.0",
    "yacs": "0.1.8",
    "tabulate": "0.9.0",
    "termcolor": "3.1.0",
    
    # --- Build ---
    "setuptools": "70.0.0",
    "cython": "3.1.4",
    "wheel": "0.45.1",
    "black": "25.1.0",
    "isort": "6.0.1",
    
    # --- Special ---
    "spconv-cu111": None 
}

def clean_version_string(ver_str):
    """Loại bỏ hậu tố như +cu124 để so sánh chính xác"""
    if '+' in ver_str:
        return ver_str.split('+')[0]
    return ver_str

def check_requirements():
    print(f"{'PACKAGE':<25} {'REQUIRED':<15} {'INSTALLED':<20} {'STATUS':<10}")
    print("=" * 75)

    all_match = True
    missing_packages = []

    for package, req_ver in REQUIRED_PACKAGES.items():
        try:
            # Lấy version hiện tại
            installed_ver_raw = importlib.metadata.version(package)
            installed_ver_clean = clean_version_string(installed_ver_raw)
            
            status = "OK"
            
            if req_ver:
                # So sánh version (đã làm sạch)
                if version.parse(installed_ver_clean) != version.parse(req_ver):
                    status = "MISMATCH"
                    all_match = False
            else:
                # Nếu không yêu cầu version cụ thể
                req_ver = "*"

            # In kết quả (Hiển thị version raw để user biết mình đang dùng CUDA nào)
            print(f"{package:<25} {req_ver:<15} {installed_ver_raw:<20} {status:<10}")

        except importlib.metadata.PackageNotFoundError:
            print(f"{package:<25} {req_ver if req_ver else '*':<15} {'Not Found':<20} {'MISSING':<10}")
            all_match = False
            missing_packages.append(package)

    print("=" * 75)
    
    if all_match:
        print("\n✅ TẤT CẢ CÁC GÓI ĐỀU ĐÚNG VERSION YÊU CẦU.")
    else:
        print("\n❌ CÓ LỖI XẢY RA:")
        if missing_packages:
            print(f"   - Thiếu gói: {', '.join(missing_packages)}")
            print("   -> Chạy: pip install -r requirements.txt")
        else:
            print("   - Sai version: Vui lòng kiểm tra lại dòng có trạng thái 'MISMATCH'.")

if __name__ == "__main__":
    check_requirements()