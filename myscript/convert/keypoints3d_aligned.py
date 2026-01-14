import numpy as np
import json
import os
from glob import glob
from itertools import combinations

# ===== Umeyama Alignment =====
def umeyama_alignment(X, Y, with_scale=True):
    n, m = X.shape
    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)
    Xc = X - mean_X
    Yc = Y - mean_Y
    C = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    if with_scale:
        var_X = (Xc**2).sum() / n
        s = np.sum(D) / var_X
    else:
        s = 1.0
    t = mean_Y - s * R @ mean_X
    return s, R, t

def transform(P, s, R, t):
    return (s * (R @ P.T).T) + t

def compute_errors(P1_aligned, P2):
    return np.linalg.norm(P1_aligned - P2, axis=1)

def find_best_anchor_combination(keypoints1, keypoints2, candidate_points, reference_point=8, min_points=3):
    """Tìm tổ hợp anchor points tốt nhất dựa trên error của reference_point (MidHip)"""
    best_error = float('inf')
    best_combination = None
    best_transform = None
    
    for size in range(min_points, len(candidate_points) + 1):
        for combo in combinations(candidate_points, size):
            try:
                X = keypoints1[list(combo)]
                Y = keypoints2[list(combo)]
                
                s, R, t = umeyama_alignment(X, Y, with_scale=True)
                
                ref_aligned = transform(keypoints1[reference_point:reference_point+1], s, R, t)
                ref_error = np.linalg.norm(ref_aligned - keypoints2[reference_point:reference_point+1])
                
                if ref_error < best_error:
                    best_error = ref_error
                    best_combination = combo
                    best_transform = (s, R, t)
                    
            except Exception as e:
                continue
    
    return best_combination, best_error, best_transform

def analyze_continuous_errors(error_log, threshold, keypoint_names, window_size, ignore_toe_heel=False):
    """Phân tích lỗi trên nhiều frame liên tục"""
    num_frames, num_points = error_log.shape
    results = []
    
    toe_heel_indices = {19, 20, 21, 22, 23, 24} if ignore_toe_heel else set()
    
    for start_frame in range(num_frames - window_size + 1):
        window = error_log[start_frame:start_frame + window_size]
        mean_errors = window.mean(axis=0)
        
        bad_points = []
        for i, error_val in enumerate(mean_errors):
            if ignore_toe_heel and i in toe_heel_indices:
                continue
            if error_val > threshold:
                bad_points.append(i)
        
        if len(bad_points) > 0:
            frame_range = f"{start_frame+1:06d}-{start_frame+window_size:06d}"
            
            error_items = []
            for pid in bad_points:
                name = keypoint_names.get(pid, f"Point{pid}")
                error_str = f"{pid}-{name}-( {mean_errors[pid]:.4f} m)"
                error_items.append(error_str)
            
            results.append({
                'frame_range': frame_range,
                'error_points': error_items
            })
    
    return results

def print_simple_error_table(error_reports, ignore_toe_heel, error_threshold, reference_cam):
    """In bảng lỗi đơn giản"""
    ignore_status = "IGNORED" if ignore_toe_heel else "INCLUDED"
    print(f"{'BẢNG ĐIỂM SAI (NGƯỠNG > ' + str(error_threshold) + 'm) - TOE/HEEL: ' + ignore_status:^120}")
    print(f"{'Reference Camera: ' + reference_cam:^120}")
    print("="*120)
    
    if not error_reports:
        print("KHÔNG CÓ ĐIỂM SAI NÀO")
        print("Chất lượng căn chỉnh: TỐT")
    else:
        for report in error_reports:
            frame_range = report['frame_range']
            error_line = "    ".join(report['error_points'])
            print(f"{frame_range}     {error_line}")
    
    print("="*120)

import re
def save_json_compact_arrays(data, filepath):
    """
    Lưu JSON với format arrays [x, y, z]
    """
    # Chuyển sang string với indent=2
    json_str = json.dumps(data, indent=2)
    
    # Pattern để tìm arrays với 3 số 
    pattern = r'\[\n\s+(-?\d+\.?\d*),\n\s+(-?\d+\.?\d*),\n\s+(-?\d+\.?\d*)\n\s+\]'
    replacement = r'[\1, \2, \3]'
    json_str = re.sub(pattern, replacement, json_str)
    
    with open(filepath, 'w') as f:
        f.write(json_str)

# ===== Main Processing =====
def process_videos(folder1, folder2, outfolder, frame_offset, error_threshold, window_size, ignore_toe_heel, reference_cam):
    
    candidate_points = [2, 5, 9, 12]
    reference_point = 8
    min_anchor_points = 3
    
    files1 = sorted(glob(os.path.join(folder1, "*.json")))
    files2 = sorted(glob(os.path.join(folder2, "*.json")))
    
    if len(files1) == 0:
        print("file does not exist in folder1")
        return
    if len(files2) == 0:
        print("file does not exist in folder2")
        return
    
    print(f"\n📹 Reference Camera: {reference_cam}")
    if reference_cam == "Camera1":
        print("   → Camera2 keypoints will be aligned to Camera1")
    else:
        print("   → Camera1 keypoints will be aligned to Camera2")
    
    if ignore_toe_heel:
        print("🦶 Toe/Heel points (19-24) will be IGNORED in error analysis")
    else:
        print("👣 ALL points will be INCLUDED in error analysis")
    
    if frame_offset == 0:
        files1_processed = files1
        files2_processed = files2
        print("⏱️  No frame offset applied")
    elif frame_offset > 0:
        files1_processed = files1
        files2_processed = files2[frame_offset:]
        print(f"⏱️  Camera2 ahead by {frame_offset} frames: skipping first {frame_offset} frames of camera2")
    else:
        files1_processed = files1[-frame_offset:]
        files2_processed = files2
        print(f"⏱️  Camera1 ahead by {-frame_offset} frames: skipping first {-frame_offset} frames of camera1")
    
    min_frames = min(len(files1_processed), len(files2_processed))
    print(f"\n📊 Files detected: Camera1={len(files1)}, Camera2={len(files2)}")
    print(f"   After sync: {min_frames} frames to process\n")
    
    if min_frames == 0:
        print("❌ No frames to process after synchronization!")
        return

    out1 = os.path.join(outfolder, "keypoints3d_1")
    out2 = os.path.join(outfolder, "keypoints3d_2")
    os.makedirs(out1, exist_ok=True)
    os.makedirs(out2, exist_ok=True)

    keypoint_names = {
        0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
        5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
        10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
        15: "REye", 16: "LEye", 17: "REar", 18: "LEar",
        19: "LBigToe", 20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel"
    }

    error_log = []

    print("⚙️  Processing frames with dynamic anchor selection...")
    for i in range(min_frames):
        if i % 100 == 0:
            print(f"  Frame {i+1}/{min_frames}...")
            
        f1 = files1_processed[i]
        f2 = files2_processed[i]
        
        frame_id1 = os.path.basename(f1).replace(".json","")
        frame_id2 = os.path.basename(f2).replace(".json","")

        with open(f1,"r") as f:
            data1 = json.load(f)
        with open(f2,"r") as f:
            data2 = json.load(f)

        keypoints1 = np.array(data1[0]["keypoints3d"])
        keypoints2 = np.array(data2[0]["keypoints3d"])

        # Xác định source và target dựa trên reference_cam
        if reference_cam == "Camera1":
            # Camera1 là gốc, align Camera2 về Camera1
            source_kp = keypoints2  # Keypoints cần transform
            target_kp = keypoints1  # Keypoints gốc (giữ nguyên)
        else:
            # Camera2 là gốc, align Camera1 về Camera2
            source_kp = keypoints1  # Keypoints cần transform
            target_kp = keypoints2  # Keypoints gốc (giữ nguyên)

        # Tìm transformation tốt nhất
        best_combo, best_error, (s, R, t) = find_best_anchor_combination(
            source_kp, target_kp, candidate_points, reference_point, min_anchor_points
        )

        # Transform source về target
        aligned_source = transform(source_kp, s, R, t)

        # Gán kết quả dựa trên reference_cam
        if reference_cam == "Camera1":
            aligned1 = keypoints1  # Camera1 giữ nguyên
            aligned2 = aligned_source  # Camera2 được align
        else:
            aligned1 = aligned_source  # Camera1 được align
            aligned2 = keypoints2  # Camera2 giữ nguyên

        aligned1_rounded = np.round(aligned1, 7)
        aligned2_rounded = np.round(aligned2, 7)
        
        data1_output = [{"id": 0, "keypoints3d": aligned1_rounded.tolist()}]
        data2_output = [{"id": 0, "keypoints3d": aligned2_rounded.tolist()}]
        
        save_json_compact_arrays(data1_output, os.path.join(out1, frame_id1+".json"))
        save_json_compact_arrays(data2_output, os.path.join(out2, frame_id2+".json"))

        # Tính error giữa hai camera sau khi align
        errors = compute_errors(aligned1[:19], aligned2[:19])
        error_log.append(errors)

    error_log = np.array(error_log)
    print(f"✅ Completed processing {min_frames} frames\n")

    print("🔍 Analyzing alignment quality...")
    results = analyze_continuous_errors(error_log, error_threshold, keypoint_names, window_size, ignore_toe_heel)
    
    print("💾 Saving aligned keypoints...")
    print_simple_error_table(results, ignore_toe_heel, error_threshold, reference_cam)
    
    # Lưu kết quả alignment keypoints3d
    error_reports_file = os.path.join(outfolder, "error_reports.json")
    save_json_compact_arrays(results, error_reports_file)

    print(f"\n📄 Error reports saved to: {error_reports_file}")
    
    error_log_file = os.path.join(outfolder, "error_log.npy")
    np.save(error_log_file, error_log)
    print(f"📊 Error log matrix saved to: {error_log_file}")
    
    metadata = {
        "total_frames": min_frames,
        "error_threshold": error_threshold,
        "window_size": window_size,
        "ignore_toe_heel": ignore_toe_heel,
        "frame_offset": frame_offset,
        "reference_camera": reference_cam,
        "candidate_anchor_points": candidate_points,
        "reference_point": reference_point,
        "keypoint_names": keypoint_names
    }
    metadata_file = os.path.join(outfolder, "alignment_metadata.json")
    save_json_compact_arrays(metadata, metadata_file)

    print(f"📋 Alignment metadata saved to: {metadata_file}")
    print(f"\n✨ Results saved to: {outfolder}/")

def main():
    """Main function"""
    folder1 = "output1/sv1p/keypoints3d"
    folder2 = "output2/sv1p/keypoints3d"
    outfolder = "aligned"
    
    if not os.path.exists(folder1):
        print(f"❌ Directory not found: {folder1}")
        return
    if not os.path.exists(folder2):
        print(f"❌ Directory not found: {folder2}")
        return
    
    print("\n" + "="*120)
    print("3D POSE ALIGNMENT - FLEXIBLE REFERENCE CAMERA MODE")
    print("="*120)
    
    # Reference camera selection
    print("\n📹 REFERENCE CAMERA SELECTION")
    print("Camera1: output1/sv1p/keypoints3d")
    print("Camera2: output2/sv1p/keypoints3d")
    print("The reference camera's keypoints will remain unchanged.")
    print("The other camera's keypoints will be aligned to match the reference.")
    while True:
        ref_input = input("\nChoose reference camera (1 or 2, default 2): ").strip()
        if not ref_input or ref_input == '2':
            reference_cam = "Camera2"
            break
        elif ref_input == '1':
            reference_cam = "Camera1"
            break
        else:
            print("Please enter '1' for Camera1 or '2' for Camera2!")
    
    # Frame offset
    print("\n⏱️  FRAME SYNCHRONIZATION")
    while True:
        try:
            offset_input = input("Frame offset (+ if cam2 ahead, - if cam1 ahead, 0 if synced): ").strip()
            frame_offset = int(offset_input)
            break
        except ValueError:
            print("Please enter a valid integer!")
    
    # Error threshold
    print("\n📏 ERROR THRESHOLD")
    while True:
        try:
            threshold_input = input("Error threshold in meters (default 0.2): ").strip()
            if not threshold_input:
                error_threshold = 0.2
            else:
                error_threshold = float(threshold_input)
                if error_threshold <= 0:
                    print("Threshold must be positive!")
                    continue
            break
        except ValueError:
            print("Please enter a valid number!")
    
    # Toe/Heel ignore option
    print("\n🦶 TOE/HEEL POINTS OPTION")
    print("Toe/Heel points (19-24) are often less accurate")
    while True:
        toe_heel_input = input("Ignore toe/heel points in error analysis? (y/n, default n): ").strip().lower()
        if not toe_heel_input or toe_heel_input in ['n', 'no']:
            ignore_toe_heel = False
            break
        elif toe_heel_input in ['y', 'yes']:
            ignore_toe_heel = True
            break
        else:
            print("Please enter 'y' for yes or 'n' for no!")
    
    window_size = 3
    
    process_videos(folder1, folder2, outfolder, frame_offset, error_threshold, window_size, ignore_toe_heel, reference_cam)

if __name__ == "__main__":
    main()