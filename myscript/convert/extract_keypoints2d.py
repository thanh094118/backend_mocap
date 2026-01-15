import numpy as np
import cv2
import os
import json
from tqdm import tqdm


def create_default_camera(img_height, img_width):
    """Tạo camera mặc định với thông số giả định"""
    focal = 1.2 * min(img_height, img_width)
    
    K = np.array([
        [focal, 0, img_width / 2],
        [0, focal, img_height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    R = np.eye(3, dtype=np.float32)
    T = np.zeros((3, 1), dtype=np.float32)
    dist = np.zeros((1, 5), dtype=np.float32)
    
    return {'K': K, 'R': R, 'T': T, 'dist': dist}


def projectPoints(X, K, R, t, Kd):
    """
    Chiếu điểm 3D lên ảnh 2D với distortion
    
    LOGIC:
    1. Chuyển điểm 3D từ world coordinates sang camera coordinates: x_cam = R @ X + t
    2. Chuẩn hóa về mặt phẳng z=1: (x/z, y/z)
    3. Áp dụng distortion (radial + tangential)
    4. Áp dụng ma trận nội tại K để ra pixel coordinates
    """
    # Bước 1: World -> Camera coordinates
    x = R @ X + t
    
    # Bước 2: Chuẩn hóa (perspective division)
    x[0:2, :] = x[0:2, :] / x[2, :]
    
    # Bước 3: Tính distortion
    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]  # r^2 = x^2 + y^2
    
    # Radial distortion: 1 + k1*r^2 + k2*r^4 + k5*r^6
    radial_dist = 1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r
    
    # Tangential distortion
    x_distorted = x[0, :] * radial_dist + \
                  2 * Kd[2] * x[0, :] * x[1, :] + \
                  Kd[3] * (r + 2 * x[0, :] * x[0, :])
    
    y_distorted = x[1, :] * radial_dist + \
                  2 * Kd[3] * x[0, :] * x[1, :] + \
                  Kd[2] * (r + 2 * x[1, :] * x[1, :])
    
    # Bước 4: Áp dụng intrinsic matrix K
    # [u, v] = K @ [x_distorted, y_distorted, 1]
    x[0, :] = K[0, 0] * x_distorted + K[0, 1] * y_distorted + K[0, 2]
    x[1, :] = K[1, 0] * x_distorted + K[1, 1] * y_distorted + K[1, 2]
    
    return x


def project_keypoints3d_to_2d(keypoints3d, camera):
    """
    Chiếu keypoints 3D lên ảnh 2D
    
    INPUT: keypoints3d shape (N, 3) - tọa độ x,y,z trong world space
    OUTPUT: keypoints2d shape (N, 2) - tọa độ u,v trong pixel space
    """
    points3d_T = keypoints3d.T  # (3, N)
    
    points2d = projectPoints(
        points3d_T,
        camera['K'],
        camera['R'],
        camera['T'],
        camera['dist'].reshape(5)
    )
    
    return points2d.T  # (N, 2)


def draw_keypoints(img, keypoints2d):
    """Vẽ keypoints và skeleton lên ảnh (Body25 format, 19 điểm)"""
    img_vis = img.copy()
    
    skeleton_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13),
        (13, 14), (0, 15), (0, 16), (15, 17), (16, 18),
    ]
    
    # Vẽ skeleton
    for pair in skeleton_pairs:
        if pair[0] >= len(keypoints2d) or pair[1] >= len(keypoints2d):
            continue
        
        kp1, kp2 = keypoints2d[pair[0]], keypoints2d[pair[1]]
        x1, y1 = int(round(kp1[0])), int(round(kp1[1]))
        x2, y2 = int(round(kp2[0])), int(round(kp2[1]))
        
        if (0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0] and
            0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]):
            cv2.line(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Vẽ keypoints
    for i, kp in enumerate(keypoints2d):
        x, y = int(round(kp[0])), int(round(kp[1]))
        
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img_vis, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(img_vis, (x, y), 7, (255, 255, 255), 2)
            cv2.putText(img_vis, str(i), (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return img_vis


def load_keypoints3d_from_json(json_path):
    """Load keypoints3d từ file JSON (chỉ lấy người đầu tiên)"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list) or len(data) == 0:
        return None
    
    person = data[0]
    
    if 'keypoints3d' not in person:
        return None
    
    kp3d = np.array(person['keypoints3d'])
    
    if kp3d.shape[0] != 25:
        return None
    
    return kp3d[:19, :]  # Chỉ lấy 19 điểm đầu


def get_frame_number(filename):
    """Trích xuất số frame từ tên file"""
    name = os.path.splitext(filename)[0]
    try:
        return int(name)
    except ValueError:
        return None


def get_frame_offset_from_user():
    """
    Hỏi user về độ lệch frame, default = 0 nếu không nhập
    
    Returns:
        int: Độ lệch frame
    """
    print("\n" + "=" * 60)
    print("THIẾT LẬP ĐỘ LỆCH FRAME")
    print("=" * 60)
    print("Hướng dẫn:")
    print("  • Nhập +N: Nếu keypoints trước ảnh N frame")
    print("           (VD: +3 = keypoint 000001 vẽ lên ảnh 000004)")
    print("  • Nhập -N: Nếu ảnh trước keypoints N frame")
    print("           (VD: -3 = keypoint 000004 vẽ lên ảnh 000001)")
    print("  • Nhấn Enter: Sử dụng mặc định (0 - đồng bộ)")
    print("=" * 60)
    
    while True:
        try:
            offset_input = input("Frame offset [default: 0]: ").strip()
            
            # Nếu không nhập gì, dùng default = 0
            if offset_input == "":
                offset = 0
            else:
                offset = int(offset_input)
            
            # Thông báo đã chọn
            if offset > 0:
                print(f"\n✓ Đã chọn offset = +{offset}")
                print(f"  → Keypoint {1:06d} sẽ vẽ lên ảnh {1+offset:06d}")
            elif offset < 0:
                print(f"\n✓ Đã chọn offset = {offset}")
                print(f"  → Keypoint {1-offset:06d} sẽ vẽ lên ảnh {1:06d}")
            else:
                print(f"\n✓ Đã chọn offset = 0 (đồng bộ)")
                print(f"  → Keypoint {1:06d} sẽ vẽ lên ảnh {1:06d}")
            
            return offset
                
        except ValueError:
            print("❌ Lỗi: Vui lòng nhập một số nguyên (VD: 3, -5, 0) hoặc Enter\n")
        except KeyboardInterrupt:
            print("\n\nĐã hủy bỏ.")
            exit(0)


def process_all_frames(images_dir, keypoints3d_dir, output_dir, frame_offset=0):
    """Xử lý tất cả các frame"""
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
    if len(image_files) == 0:
        print(f"Không tìm thấy file ảnh trong {images_dir}")
        return
    
    # Tạo camera
    first_img = cv2.imread(os.path.join(images_dir, image_files[0]))
    height, width = first_img.shape[:2]
    camera = create_default_camera(height, width)
    
    # Xử lý frames
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for img_file in tqdm(image_files, desc="Processing frames"):
        img_frame_num = get_frame_number(img_file)
        if img_frame_num is None:
            fail_count += 1
            continue
        
        kp_frame_num = img_frame_num + frame_offset
        kp_filename = f"{kp_frame_num:06d}.json"
        json_path = os.path.join(keypoints3d_dir, kp_filename)
        
        if not os.path.exists(json_path):
            skip_count += 1
            continue
        
        img = cv2.imread(os.path.join(images_dir, img_file))
        if img is None:
            fail_count += 1
            continue
        
        keypoints3d = load_keypoints3d_from_json(json_path)
        if keypoints3d is None:
            fail_count += 1
            continue
        
        keypoints2d = project_keypoints3d_to_2d(keypoints3d, camera)
        img_vis = draw_keypoints(img, keypoints2d)
        
        text = f"Image: {img_frame_num:06d} | Keypoint: {kp_frame_num:06d}"
        cv2.putText(img_vis, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        vis_path = os.path.join(vis_dir, f"{img_frame_num:06d}.jpg")
        cv2.imwrite(vis_path, img_vis)
        
        success_count += 1
    
    print(f"Thành công: {success_count}/{len(image_files)}")
    print(f"Bỏ qua (không có keypoint tương ứng): {skip_count}/{len(image_files)}")
    print(f"Thất bại: {fail_count}/{len(image_files)}")


def make_video(output_dir, fps=30):
    """Tạo video từ ảnh"""
    if os.system("ffmpeg -version > /dev/null 2>&1") != 0:
        print("Lỗi: ffmpeg chưa được cài. Cài bằng: sudo apt install ffmpeg")
        return
    
    vis_dir = os.path.join(output_dir, 'visualizations')
    video_dir = os.path.join(output_dir, 'video')
    os.makedirs(video_dir, exist_ok=True)
    
    import glob
    if len(glob.glob(os.path.join(vis_dir, '*.jpg'))) == 0:
        print(f"Không có ảnh trong {vis_dir}")
        return
    
    video_path = os.path.join(video_dir, 'visualization.mp4')
    cmd = (
        f'ffmpeg -y -r {fps} -i "{vis_dir}/%06d.jpg" '
        f'-vf scale="2*ceil(iw/2):2*ceil(ih/2)" '
        f'-pix_fmt yuv420p -vcodec libx264 -r {fps} "{video_path}" -loglevel quiet'
    )
    
    os.system(cmd)
    
    if os.path.exists(video_path):
        print(f"Video đã lưu: {video_path}")


def main():
    images_dir = "data/images/images"
    keypoints3d_dir = "output2/sv1p/keypoints3d" #"aligned/keypoints3d_2"
    output_dir = "output_keypoints2d"
    
    if not os.path.exists(images_dir):
        print(f"Lỗi: Không tìm thấy thư mục {images_dir}")
        return
    
    if not os.path.exists(keypoints3d_dir):
        print(f"Lỗi: Không tìm thấy thư mục {keypoints3d_dir}")
        return
    
    # Hỏi frame offset (default = 0)
    frame_offset = get_frame_offset_from_user()
    
    # Xử lý ảnh
    process_all_frames(images_dir, keypoints3d_dir, output_dir, frame_offset)
    
    # Hỏi có muốn tạo video không
    make_video_choice = input("\nTạo video? (y/n): ").strip().lower()
    if make_video_choice == 'y':
        make_video(output_dir, fps=30)


if __name__ == "__main__":
    main()