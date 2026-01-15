import cv2
import numpy as np
import os
import glob
import json
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN ---
CAM1_FOLDER = 'output/camera1/sv1p/render'
CAM2_FOLDER = 'output/camera2/sv1p/render'
OUTPUT_FILE = 'output/occlusion.json' # Đã sửa đường dẫn output

# --- THÔNG SỐ KỸ THUẬT ---
TOLERANCE = 1
MIN_PIXEL_THRESHOLD = 0

# --- CẤU HÌNH MÀU ---
PART_DEFINITIONS = {
    "Left Leg": { "color_rgb": np.array([1.0, 0.0, 1.0]) },   # Magenta
    "Right Leg": { "color_rgb": np.array([0.0, 1.0, 1.0]) },  # Cyan
    "Left Arm": { "color_rgb": np.array([1.0, 1.0, 0.0]) },   # Blue (theo logic cũ của bạn)
    "Right Arm": { "color_rgb": np.array([0.0, 0.0, 1.0]) }  # Yellow (theo logic cũ của bạn)
}

def detect_occluded_parts(img_path):
    """Trả về danh sách các bộ phận bị che trong 1 ảnh"""
    img = cv2.imread(img_path)
    if img is None: return []

    occluded_list = []
    
    for part_name, data in PART_DEFINITIONS.items():
        # RGB -> BGR
        target_rgb = (data["color_rgb"] * 255).astype(int)
        target_bgr = np.array([target_rgb[2], target_rgb[1], target_rgb[0]])
        
        # Masking
        lower = np.clip(target_bgr - TOLERANCE, 0, 255)
        upper = np.clip(target_bgr + TOLERANCE, 0, 255)
        mask = cv2.inRange(img, lower, upper)
        
        count = cv2.countNonZero(mask)
        
        if count <= MIN_PIXEL_THRESHOLD:
            occluded_list.append(part_name)
            
    return occluded_list

def get_filenames(folder_path):
    """Lấy danh sách tên file (basename) trong folder"""
    extensions = ['*.jpg', '*.png', '*.jpeg']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    # Chỉ lấy tên file (ví dụ: '0001.jpg') để so sánh
    return set([os.path.basename(f) for f in files])

def process_and_aggregate():
    # 1. Tạo thư mục output nếu chưa có
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Khởi tạo data
    final_data = {
        "camera1": { key: [] for key in PART_DEFINITIONS.keys() },
        "camera2": { key: [] for key in PART_DEFINITIONS.keys() }
    }

    # 3. Lấy danh sách file và tính toán số lượng chung
    files_cam1_set = get_filenames(CAM1_FOLDER)
    files_cam2_set = get_filenames(CAM2_FOLDER)

    # Lấy danh sách file tồn tại ở CẢ 2 folder (Intersection)
    # Cách này an toàn hơn là cắt list, vì nó đảm bảo cùng tên file (cùng timestamp/frame)
    common_files = sorted(list(files_cam1_set & files_cam2_set))

    if not common_files:
        print("Không tìm thấy file chung nào giữa 2 thư mục hoặc thư mục rỗng.")
        return

    print(f"Cam 1 total: {len(files_cam1_set)}")
    print(f"Cam 2 total: {len(files_cam2_set)}")
    print(f"Processing {len(common_files)} common images...")

    # 4. Vòng lặp xử lý
    for filename in tqdm(common_files, desc="Processing"):
        path1 = os.path.join(CAM1_FOLDER, filename)
        path2 = os.path.join(CAM2_FOLDER, filename)
        
        # --- XỬ LÝ CAMERA 1 ---
        occluded_parts_1 = detect_occluded_parts(path1)
        for part in occluded_parts_1:
            final_data["camera1"][part].append(filename)

        # --- XỬ LÝ CAMERA 2 ---
        # Không cần check exists nữa vì đã lọc qua common_files
        occluded_parts_2 = detect_occluded_parts(path2)
        for part in occluded_parts_2:
            final_data["camera2"][part].append(filename)

    # 5. Lưu file JSON
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        
        print(f"Successfully saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    process_and_aggregate()