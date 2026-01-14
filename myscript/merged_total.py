import json
import os
from pathlib import Path
from typing import Dict, List, Set

def get_bone_from_limb(limb: str, part: str) -> List[str]:
    """
    Convert limb name to bone pairs.
    
    Args:
        limb: "Left Arm", "Right Arm", "Left Leg", "Right Leg"
        part: "upper" or "lower" to specify which bone pair
    
    Returns:
        List of joint names forming the bone
    """
    limb_lower = limb.lower()
    
    if "arm" in limb_lower:
        if "left" in limb_lower:
            if part == "upper":
                return ["left_shoulder", "left_elbow"]
            else:  # lower
                return ["left_elbow", "left_hand"]
        else:  # right
            if part == "upper":
                return ["right_shoulder", "right_elbow"]
            else:  # lower
                return ["right_elbow", "right_hand"]
    else:  # leg
        if "left" in limb_lower:
            if part == "upper":
                return ["left_hip", "left_knee"]
            else:  # lower
                return ["left_knee", "left_ankle"]
        else:  # right
            if part == "upper":
                return ["right_hip", "right_knee"]
            else:  # lower
                return ["right_knee", "right_ankle"]

def get_all_bones_from_limb(limb: str) -> List[List[str]]:
    """Get both upper and lower bone pairs for a limb."""
    return [
        get_bone_from_limb(limb, "upper"),
        get_bone_from_limb(limb, "lower")
    ]

def format_bone_name(bone: List[str]) -> str:
    """Format bone pair as 'left/right_joint1_joint2'."""
    # Extract prefix (left_ or right_) and joint names
    joint1_full = bone[0]
    joint2_full = bone[1]
    
    # Get prefix (left_ or right_)
    if joint1_full.startswith("left_"):
        prefix = "left_"
        joint1 = joint1_full[5:]  # Remove "left_"
        joint2 = joint2_full[5:]  # Remove "left_"
    elif joint1_full.startswith("right_"):
        prefix = "right_"
        joint1 = joint1_full[6:]  # Remove "right_"
        joint2 = joint2_full[6:]  # Remove "right_"
    else:
        joint1 = joint1_full
        joint2 = joint2_full
        prefix = ""
    
    return f"{prefix}{joint1}_{joint2}"

def process_pose_with_occlusion(pose_folder: str, occlusion_file: str, output_folder: str = None):
    """
    Process pose JSON files and merge with occlusion information.
    
    Args:
        pose_folder: Folder containing pose JSON files
        occlusion_file: Path to occlusion.json file
        output_folder: Output folder (if None, saves to same folder as pose_folder)
    """
    # Load occlusion data
    with open(occlusion_file, 'r') as f:
        occlusion_data = json.load(f)
    
    # Create output folder if needed
    if output_folder is None:
        output_folder = pose_folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all JSON files in pose folder
    pose_files = sorted(Path(pose_folder).glob('*.json'))
    
    for pose_file in pose_files:
        # Get frame name (e.g., "000006.jpg" from "000006.json")
        frame_name = pose_file.stem + '.jpg'
        
        # Load pose data
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
        
        # Determine which bones are wrong in each camera
        wrong_bones_per_camera = {}
        
        for camera, limbs in occlusion_data.items():
            wrong_bones = set()
            
            for limb, frame_list in limbs.items():
                if frame_name in frame_list:
                    # This limb is occluded/wrong in this frame
                    bones = get_all_bones_from_limb(limb)
                    for bone in bones:
                        wrong_bones.add(tuple(bone))
            
            wrong_bones_per_camera[camera] = wrong_bones
        
        # Determine camera-specific correctness
        cameras = list(occlusion_data.keys())
        if len(cameras) == 2:
            camera1, camera2 = cameras[0], cameras[1]
            
            # Bones wrong only in camera1 (correct only in camera2)
            only_camera2_correct = wrong_bones_per_camera[camera1] - wrong_bones_per_camera[camera2]
            
            # Bones wrong only in camera2 (correct only in camera1)
            only_camera2_wrong = wrong_bones_per_camera[camera2] - wrong_bones_per_camera[camera1]
            
            # Format bone names for output
            pose_data[f"only_{camera2}_correct"] = sorted([
                format_bone_name(list(bone)) for bone in only_camera2_correct
            ])
            pose_data[f"only_{camera2}_wrong"] = sorted([
                format_bone_name(list(bone)) for bone in only_camera2_wrong
            ])
        
        # Save merged data with custom formatting
        output_file = Path(output_folder) / pose_file.name
        save_pose_json(pose_data, output_file)
        
        print(f"Processed: {pose_file.name}")

def save_pose_json(data: dict, filepath: str):
    """
    Save pose data with custom formatting:
    - Coordinates on single line
    - Normal formatting for metadata fields
    """
    with open(filepath, 'w') as f:
        f.write('{\n')
        
        cameras = [k for k in data.keys() if k.startswith('camera')]
        metadata_keys = [k for k in data.keys() if not k.startswith('camera')]
        
        # Write camera data
        for i, camera in enumerate(cameras):
            f.write(f'    "{camera}": {{\n')
            camera_data = data[camera]
            joints = list(camera_data.keys())
            
            for j, joint in enumerate(joints):
                coords = camera_data[joint]
                coords_str = json.dumps(coords)
                
                if j < len(joints) - 1:
                    f.write(f'        "{joint}": {coords_str},\n')
                else:
                    f.write(f'        "{joint}": {coords_str}\n')
            
            if i < len(cameras) - 1 or metadata_keys:
                f.write('    },\n')
            else:
                f.write('    }\n')
        
        # Write metadata (only_camera2_correct, only_camera2_wrong)
        for i, key in enumerate(metadata_keys):
            value = json.dumps(data[key], indent=4)
            
            if i < len(metadata_keys) - 1:
                f.write(f'    "{key}": {value},\n')
            else:
                f.write(f'    "{key}": {value}\n')
        
        f.write('}\n')

if __name__ == "__main__":
    # Example usage
    pose_folder = "output/merged_poses"  # Folder containing pose JSON files
    occlusion_file = "output/occlusion.json"  # Path to occlusion.json
    output_folder = "output/merged_total"  # Optional: specify different output folder
    
    process_pose_with_occlusion(pose_folder, occlusion_file, output_folder)