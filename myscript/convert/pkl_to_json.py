import pickle
import json
import argparse
import sys
from pathlib import Path

def try_load_pickle(input_file):
    """Thử nhiều cách load file pickle"""
    
    # Cách 1: Load bình thường
    try:
        with open(input_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e1:
        print(f"  Cách 1 thất bại: {e1}")
    
    # Cách 2: Load với encoding latin1
    try:
        with open(input_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e2:
        print(f"  Cách 2 thất bại: {e2}")
    
    # Cách 3: Load với encoding bytes
    try:
        with open(input_file, 'rb') as f:
            return pickle.load(f, encoding='bytes')
    except Exception as e3:
        print(f"  Cách 3 thất bại: {e3}")
    
    # Cách 4: Thử dill (nếu có)
    try:
        import dill
        with open(input_file, 'rb') as f:
            return dill.load(f)
    except ImportError:
        print("  Cách 4: Không có thư viện dill")
    except Exception as e4:
        print(f"  Cách 4 thất bại: {e4}")
    
    # Cách 5: Thử joblib (nếu có)
    try:
        import joblib
        return joblib.load(input_file)
    except ImportError:
        print("  Cách 5: Không có thư viện joblib")
    except Exception as e5:
        print(f"  Cách 5 thất bại: {e5}")
    
    # Cách 6: Thử torch (nếu có)
    try:
        import torch
        return torch.load(input_file, map_location='cpu')
    except ImportError:
        print("  Cách 6: Không có thư viện torch")
    except Exception as e6:
        print(f"  Cách 6 thất bại: {e6}")
    
    return None

def convert_to_serializable(obj):
    """Chuyển đổi object sang dạng có thể serialize JSON"""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except:
            return str(obj)
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return obj

def convert_pkl_to_json(input_file, output_file=None):
    """
    Chuyển đổi file .pkl sang .json
    
    Args:
        input_file: Đường dẫn file .pkl đầu vào
        output_file: Đường dẫn file .json đầu ra (tùy chọn)
    """
    try:
        # Kiểm tra file tồn tại
        if not Path(input_file).exists():
            print(f"✗ Lỗi: Không tìm thấy file '{input_file}'", file=sys.stderr)
            sys.exit(1)
        
        # Kiểm tra kích thước file
        file_size = Path(input_file).stat().st_size
        print(f"📁 Kích thước file: {file_size:,} bytes")
        
        # Đọc file pickle với nhiều cách
        print("🔄 Đang thử load file...")
        data = try_load_pickle(input_file)
        
        if data is None:
            print("\n✗ Không thể load file bằng bất kỳ phương thức nào!", file=sys.stderr)
            print("\n💡 Gợi ý:")
            print("  - File có thể không phải là pickle/pkl")
            print("  - File có thể bị hỏng")
            print("  - Thử cài đặt: pip install dill joblib")
            sys.exit(1)
        
        print(f"✓ Load thành công! Kiểu dữ liệu: {type(data).__name__}")
        
        # Chuyển đổi sang dạng có thể serialize
        print("🔄 Đang chuyển đổi sang JSON...")
        try:
            serializable_data = convert_to_serializable(data)
        except Exception as e:
            print(f"⚠ Cảnh báo khi chuyển đổi: {e}")
            serializable_data = data
        
        # Tạo tên file output nếu không được cung cấp
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.with_suffix('.json')
        
        # Ghi ra file JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2, default=str)
        
        output_size = Path(output_file).stat().st_size
        print(f"\n✓ Chuyển đổi thành công!")
        print(f"  Input:  {input_file} ({file_size:,} bytes)")
        print(f"  Output: {output_file} ({output_size:,} bytes)")
        
    except Exception as e:
        print(f"\n✗ Lỗi không xác định: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Chuyển đổi file .pkl sang .json (hỗ trợ nhiều định dạng)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python convert.py -i data.pkl
  python convert.py -i data.pkl -o output.json
  python convert.py --input data.pkl --output result.json

Script hỗ trợ:
  - pickle (standard)
  - dill
  - joblib
  - torch (PyTorch)
  - numpy arrays
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='File .pkl đầu vào'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='File .json đầu ra (mặc định: tên file input với đuôi .json)'
    )
    
    args = parser.parse_args()
    
    convert_pkl_to_json(args.input, args.output)

if __name__ == '__main__':
    main()