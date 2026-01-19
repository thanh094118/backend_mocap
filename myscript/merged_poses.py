import json
import sys
import os
from pathlib import Path

# Mapping từ ID sang tên keypoint
KEYPOINT_MAPPING = {
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_hand",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_hand",
    9: "right_hip",
    10: "right_knee",
    11: "right_ankle",
    12: "left_hip",
    13: "left_knee",
    14: "left_ankle",
    1: "neck"
}

def process_json_file(file_path):
    """
    Đọc file JSON và trích xuất các keypoints cần thiết
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Lấy frame đầu tiên (index 0)
        frame_data = data[0]
        keypoints3d = frame_data['keypoints3d']
        
        # Trích xuất các keypoints theo ID cần thiết
        result = {}
        for idx, name in KEYPOINT_MAPPING.items():
            result[name] = keypoints3d[idx]
        
        return result
    
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None

def write_merged_json(output_path, camera1_data, camera2_data):
    """
    Ghi file JSON với format compact cho mảng
    """
    output_data = {
        "camera1": camera1_data,
        "camera2": camera2_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('{\n')
        
        for cam_idx, (cam_name, cam_data) in enumerate(output_data.items()):
            f.write(f'  "{cam_name}": {{\n')
            
            keypoint_items = list(cam_data.items())
            for kp_idx, (kp_name, coords) in enumerate(keypoint_items):
                coords_str = json.dumps(coords)
                comma = ',' if kp_idx < len(keypoint_items) - 1 else ''
                f.write(f'    "{kp_name}": {coords_str}{comma}\n')
            
            comma = ',' if cam_idx < len(output_data) - 1 else ''
            f.write(f'  }}{comma}\n')
        
        f.write('}\n')

def merge_folders(folder1_path, folder2_path, output_folder_path):
    """
    Xử lý tất cả các file JSON trong 2 folder và gộp vào folder output
    """
    # Tạo folder output nếu chưa tồn tại
    output_folder = Path(output_folder_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách file JSON từ folder 1
    folder1 = Path(folder1_path)
    folder2 = Path(folder2_path)
    
    if not folder1.exists():
        print(f"Lỗi: Folder {folder1_path} không tồn tại")
        sys.exit(1)
    
    if not folder2.exists():
        print(f"Lỗi: Folder {folder2_path} không tồn tại")
        sys.exit(1)
    
    # Lấy tất cả file JSON từ folder 1
    json_files = sorted([f for f in folder1.glob('*.json')])
    
    if not json_files:
        print(f"Cảnh báo: Không tìm thấy file JSON nào trong {folder1_path}")
        sys.exit(1)
    
    print(f"Tìm thấy {len(json_files)} file JSON để xử lý")
    
    success_count = 0
    error_count = 0
    
    # Xử lý từng cặp file
    for json_file in json_files:
        file_name = json_file.name
        file2_path = folder2 / file_name
        
        # Kiểm tra file tương ứng có tồn tại trong folder 2
        if not file2_path.exists():
            error_count += 1
            continue
        
        # Xử lý 2 file
        camera1_data = process_json_file(json_file)
        camera2_data = process_json_file(file2_path)
        
        if camera1_data is None or camera2_data is None:
            error_count += 1
            continue
        
        # Ghi file output
        output_file = output_folder / file_name
        try:
            write_merged_json(output_file, camera1_data, camera2_data)
            success_count += 1
        except Exception as e:
            print(f"✗ Lỗi khi ghi file {file_name}: {e}")
            error_count += 1
    
    print(f"\n{'='*50}")
    print(f"Hoàn thành!")
    print(f"Thành công: {success_count} file")
    print(f"Lỗi: {error_count} file")
    print(f"Kết quả đã lưu vào: {output_folder_path}")

# Cấu hình đường dẫn cố định
CAMERA1_FOLDER = "output/camera1/sv1p/keypoints3d"
CAMERA2_FOLDER = "output/camera2/sv1p/keypoints3d"
OUTPUT_FOLDER = "output/merged_poses"

if __name__ == "__main__":
    print(f"Camera 1 folder: {CAMERA1_FOLDER}")
    print(f"Camera 2 folder: {CAMERA2_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"{'='*50}\n")
    
    merge_folders(CAMERA1_FOLDER, CAMERA2_FOLDER, OUTPUT_FOLDER)
