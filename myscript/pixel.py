import json
import os
from pathlib import Path

# Cấu hình đường dẫn
CAM1_FOLDER = 'output/camera1/sv1p/confident'
CAM2_FOLDER = 'output/camera2/sv1p/confident'
OUTPUT_FOLDER = 'output/confident'

# Bảng ánh xạ joint_id sang tên bone
JOINT_MAPPING = {
    3: "right_shoulder_elbow",
    4: "right_elbow_hand",
    6: "left_shoulder_elbow",
    7: "left_elbow_hand",
    10: "right_hip_knee",
    11: "right_knee_ankle",
    13: "left_hip_knee",
    14: "left_knee_ankle"
}

def read_json_file(filepath):
    """Đọc file JSON và trả về dictionary với bone name làm key"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        result = {}
        for item in data:
            joint_id = item['joint_id']
            if joint_id in JOINT_MAPPING:
                bone_name = JOINT_MAPPING[joint_id]
                result[bone_name] = item
        return result
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(f"Lỗi đọc JSON: {filepath}")
        return {}

def merge_camera_data(cam1_data, cam2_data):
    """Gộp dữ liệu từ 2 camera"""
    result = {
        "only_camera2_correct": [],
        "only_camera2_wrong": []
    }
    
    # Lấy tất cả bone names từ cả 2 camera
    all_bones = set(cam1_data.keys()) | set(cam2_data.keys())
    
    for bone_name in sorted(all_bones):
        # Nếu chỉ có cam1
        if bone_name in cam1_data and bone_name not in cam2_data:
            result["only_camera2_correct"].append(bone_name)
        # Nếu chỉ có cam2
        elif bone_name not in cam1_data and bone_name in cam2_data:
            result["only_camera2_wrong"].append(bone_name)
        # Nếu cả 2 đều có, ưu tiên theo logic của bạn
        # (có thể điều chỉnh logic này nếu cần)
        else:
            result["only_camera2_correct"].append(bone_name)
    
    return result

def process_files():
    """Xử lý tất cả các file JSON"""
    # Tạo thư mục output nếu chưa có
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Lấy danh sách tất cả file JSON từ cả 2 thư mục
    cam1_files = set()
    cam2_files = set()
    
    if os.path.exists(CAM1_FOLDER):
        cam1_files = {f for f in os.listdir(CAM1_FOLDER) if f.endswith('.json')}
    
    if os.path.exists(CAM2_FOLDER):
        cam2_files = {f for f in os.listdir(CAM2_FOLDER) if f.endswith('.json')}
    
    all_files = cam1_files | cam2_files
    
    print(f"Tìm thấy {len(all_files)} file cần xử lý")
    
    # Xử lý từng file
    for filename in sorted(all_files):
        cam1_path = os.path.join(CAM1_FOLDER, filename)
        cam2_path = os.path.join(CAM2_FOLDER, filename)
        
        # Đọc dữ liệu từ cả 2 camera
        cam1_data = read_json_file(cam1_path)
        cam2_data = read_json_file(cam2_path)
        
        # Gộp dữ liệu
        merged_data = merge_camera_data(cam1_data, cam2_data)
        
        # Ghi file output
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"Đã xử lý: {filename}")
    
    print(f"\nHoàn thành! Kết quả được lưu tại: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    process_files()
