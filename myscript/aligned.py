import numpy as np
import json
import os
import glob
import pdb

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn folder chứa file JSON đầu vào
INPUT_FOLDER = 'output/merged_total' 
# Đường dẫn folder chứa file JSON kết quả đầu ra
OUTPUT_FOLDER = 'output/output_final'

def umeyama(P, Q):
    """Tính toán ma trận xoay R để đồng nhất hướng giữa P và Q"""
    n = P.shape[0]
    mu_p = P.mean(axis=0)
    mu_q = Q.mean(axis=0)
    P_c = P - mu_p
    Q_c = Q - mu_q
    H = (Q_c.T @ P_c) / n
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return R

def get_angle_between_vectors(v1, v2):
    """Tính góc lệch (độ) giữa hai vector"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0
    
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def calculate_joint_angle(p1, p2, p3):
    """Tính góc tại khớp p2 được tạo bởi 3 điểm p1-p2-p3"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    return get_angle_between_vectors(v1, v2)

def check_joint_angles(cam1, cam2, R_global, angle_threshold=30.0):
    """Kiểm tra góc khớp giữa các cặp đoạn xương liên tiếp"""
    divergent_joints = []
    
    # Định nghĩa các khớp cần kiểm tra: (tên, điểm đầu, điểm khớp, điểm cuối)
    joint_definitions = [
        ("right_elbow", "right_shoulder", "right_elbow", "right_hand"),
        ("left_elbow", "left_shoulder", "left_elbow", "left_hand"),
        ("right_knee", "right_hip", "right_knee", "right_ankle"),
        ("left_knee", "left_hip", "left_knee", "left_ankle")
    ]
    
    for joint_name, start_key, joint_key, end_key in joint_definitions:
        # Kiểm tra xem cả 3 điểm có tồn tại ở cả 2 camera không
        if all(k in cam1 for k in [start_key, joint_key, end_key]) and \
           all(k in cam2 for k in [start_key, joint_key, end_key]):
            
            # Tính góc khớp cho camera 1
            angle_cam1 = calculate_joint_angle(
                cam1[start_key],
                cam1[joint_key],
                cam1[end_key]
            )
            
            # Tính góc khớp cho camera 2
            angle_cam2 = calculate_joint_angle(
                cam2[start_key],
                cam2[joint_key],
                cam2[end_key]
            )
            
            # Tính độ lệch góc
            angle_diff = abs(angle_cam1 - angle_cam2)
            
            # Nếu lệch quá ngưỡng thì báo sai
            if angle_diff > angle_threshold:
                divergent_joints.append({
                    "joint": joint_name,
                    "camera1_angle": round(float(angle_cam1), 2),
                    "camera2_angle": round(float(angle_cam2), 2),
                    "deviation": round(float(angle_diff), 2),
                    "threshold": angle_threshold,
                    "status": "misaligned"
                })
    
    # Sắp xếp theo độ lệch từ cao xuống thấp
    divergent_joints = sorted(divergent_joints, key=lambda x: x['deviation'], reverse=True)
    return divergent_joints

def find_anomalies(diff_bone_angle_segment, ignore = {}, always = {}):
    lst = []
    minValue = 180
    for k in diff_bone_angle_segment.keys():
        if not k in ignore:
            lst.append(diff_bone_angle_segment[k])
        if k in always:
            if minValue > abs(diff_bone_angle_segment[k]):
                minValue = abs(diff_bone_angle_segment[k])
    
    # Xử lý trường hợp danh sách rỗng
    if not lst:
        return []

    # 1. Tính độ lệch tuyệt đối giữa các cặp
    diffs = np.abs(np.array(lst))
    
    # 2. Tính toán các mốc tứ phân vị
    q1 = np.percentile(diffs, 25)
    q3 = np.percentile(diffs, 75)
    iqr = q3 - q1
    
    # 3. Xác định ngưỡng bất thường
    upper_bound = min(q3 + 0 * iqr, minValue)
    print(f"  -> Upper bound (IQR): {round(float(upper_bound), 2)}, Min Value: {round(float(minValue), 2)}")

    anomalies = []
    for key, value in diff_bone_angle_segment.items():
        if (value >= upper_bound and not (key in ignore)) or (key in always):
            anomalies.append({
                "segment" : key,
                "deviation": round(float(value), 4),
                "threshold": round(float(upper_bound), 4),
                "status": "misaligned"
            })
        
    x = sorted(diff_bone_angle_segment.items(), key = lambda item: item[1], reverse = True)
    
    # Sắp xếp các điểm bất thường từ nặng nhất đến nhẹ nhất
    anomalies = sorted(anomalies, key=lambda x: x['deviation'], reverse=True)
    return anomalies

def analyze_pose_difference(file_path, angle_threshold=15.0, joint_angle_threshold=50.0):
    if not os.path.exists(file_path):
        return json.dumps({"status": "error", "message": "File not found."}, indent=4)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        cam1 = data.get("camera1", {})
        cam2 = data.get("camera2", {})
        ignore = {}
        always = {}
        try:
            ignore = data.get("only_camera2_correct", {})
            always = data.get("only_camera2_wrong", {})
        except:
            pass

        # 1. Tìm ma trận xoay (Ưu tiên dùng Tam giác thân: 2 Vai + MidHip)
        torso_req = ["right_shoulder", "left_shoulder", "left_hip", "right_hip"]
        has_torso_c1 = all(k in cam1 for k in torso_req)
        has_torso_c2 = all(k in cam2 for k in torso_req)

        p1 = []
        p2 = []

        if has_torso_c1 and has_torso_c2:
            # Tính Mid-Hip cho Cam 1
            mid_hip_c1 = (np.array(cam1["left_hip"]) + np.array(cam1["right_hip"])) / 2.0
            p1 = np.array([
                cam1["right_shoulder"], 
                cam1["left_shoulder"], 
                mid_hip_c1
            ])

            # Tính Mid-Hip cho Cam 2
            mid_hip_c2 = (np.array(cam2["left_hip"]) + np.array(cam2["right_hip"])) / 2.0
            p2 = np.array([
                cam2["right_shoulder"], 
                cam2["left_shoulder"], 
                mid_hip_c2
            ])

        else:
            # FALLBACK: Cách cũ (Dùng tất cả điểm chung)
            common_keys = [k for k in cam1 if k in cam2]
            if len(common_keys) < 3:
                return json.dumps({"status": "error", "message": "Too few points."}, indent=4)
            
            p1 = np.array([cam1[k] for k in common_keys])
            p2 = np.array([cam2[k] for k in common_keys])

        # Tính toán R dựa trên p1, p2 đã chọn
        R_global = umeyama(p1, p2)

        # 2. Định nghĩa các đoạn nối (Xương)
        bone_definitions = [
            ("right_shoulder_elbow", "right_shoulder", "right_elbow"),
            ("right_elbow_hand", "right_elbow", "right_hand"),
            ("left_shoulder_elbow", "left_shoulder", "left_elbow"),
            ("left_elbow_hand", "left_elbow", "left_hand"),
            ("right_hip_knee", "right_hip", "right_knee"),
            ("right_knee_ankle", "right_knee", "right_ankle"),
            ("left_hip_knee", "left_hip", "left_knee"),
            ("left_knee_ankle", "left_knee", "left_ankle"),
            ("right_shoulder_hip", "right_hip", "right_shoulder"),
            ("left_shoulder_hip", "left_hip", "left_shoulder"),
            ("shoulders", "left_shoulder", "right_shoulder"),
            ("hips", "left_hip", "right_hip")
        ]

        diff_bone_angle_segment = {}

        # 3. Duyệt qua từng đoạn xương để so sánh hướng
        for bone_name, start_key, end_key in bone_definitions:
            if all(k in cam1 for k in [start_key, end_key]) and \
               all(k in cam2 for k in [start_key, end_key]):
                
                # Vector xương ở Cam 1 (sau đó xoay sang hệ của Cam 2)
                v1 = np.array(cam1[end_key]) - np.array(cam1[start_key])
                v1_aligned = R_global @ v1
                
                # Vector xương ở Cam 2
                v2 = np.array(cam2[end_key]) - np.array(cam2[start_key])
                
                # Tính độ lệch hướng
                angle_diff = get_angle_between_vectors(v1_aligned, v2)
                diff_bone_angle_segment[bone_name] = round(float(angle_diff), 2)
        
        anomalies = find_anomalies(diff_bone_angle_segment, ignore, always)
        
        # 4. KIỂM TRA GÓC KHỚP
        divergent_joints = check_joint_angles(cam1, cam2, R_global, joint_angle_threshold)

        # 5. Xuất kết quả JSON
        output = {
            "status": "success",
            "summary": {
                "total_segments_checked": len(bone_definitions),
                "divergent_segments_count": len(anomalies),
                "divergent_joints_count": len(divergent_joints),
                "pose_consistency_score": round(max(0, 100 - (len(anomalies)/len(bone_definitions)*100)), 2)
            },
            "divergent_segments": anomalies,
            "divergent_with_2_segments": divergent_joints
        }

        return json.dumps(output, indent=4)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, indent=4)

if __name__ == "__main__":
    # 1. Kiểm tra folder đầu vào
    if not os.path.exists(INPUT_FOLDER):
        print(f"LỖI: Không tìm thấy thư mục input: {INPUT_FOLDER}")
        print("Vui lòng tạo thư mục và bỏ file JSON vào đó.")
        exit()

    # 2. Tạo folder đầu ra nếu chưa có
    if not os.path.exists(OUTPUT_FOLDER):
        try:
            os.makedirs(OUTPUT_FOLDER)
            print(f"Đã tạo thư mục output: {OUTPUT_FOLDER}")
        except Exception as e:
            print(f"Lỗi không thể tạo thư mục output: {e}")
            exit()

    # 3. Lấy danh sách file JSON trong input
    target_files = glob.glob(os.path.join(INPUT_FOLDER, '*.json'))
    
    # Lọc bỏ các file output (nếu lỡ lưu chung folder) để tránh loop vô hạn
    target_files = [f for f in target_files if "pose_analysis_" not in os.path.basename(f)]

    if not target_files:
        print(f"Không tìm thấy file .json nào trong: {INPUT_FOLDER}")
    else:
        print(f"Tìm thấy {len(target_files)} file trong '{INPUT_FOLDER}'. Bắt đầu xử lý...\n")

        for json_file in target_files:
            filename = os.path.basename(json_file)
            print(f"Processing: {filename}...")
            
            # Xử lý
            result_json = analyze_pose_difference(json_file)
            
            # Tạo đường dẫn output
            output_name = "pose_analysis_" + filename
            output_path = os.path.join(OUTPUT_FOLDER, output_name)
            
            # Lưu file
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result_json)
                print(f" -> Saved: {output_path}")
            except Exception as e:
                print(f" -> Error saving result: {e}")
        
        print("\nHoàn tất xử lý tất cả file.")