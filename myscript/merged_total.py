import json
import os
from pathlib import Path
from typing import Dict, List

def process_pose_with_confident(pose_folder: str, confident_folder: str, output_folder: str = None):
    """
    Process pose JSON files and merge with confident information.
    
    Args:
        pose_folder: Folder containing pose JSON files (e.g., "output/merged_poses")
        confident_folder: Folder containing confident JSON files (e.g., "output/confident")
        output_folder: Output folder (if None, saves to pose_folder with "_final" suffix)
    """
    # Create output folder if needed
    if output_folder is None:
        output_folder = pose_folder + "_final"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all JSON files in pose folder
    pose_files = sorted(Path(pose_folder).glob('*.json'))
    
    if not pose_files:
        print(f"Warning: No JSON files found in {pose_folder}")
        return
    
    processed_count = 0
    
    for pose_file in pose_files:
        # Load pose data
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
        
        # Try to load corresponding confident file
        confident_file = Path(confident_folder) / pose_file.name
        
        if os.path.exists(confident_file):
            try:
                with open(confident_file, 'r') as f:
                    confident_data = json.load(f)
                
                # Add confident data to pose data
                if "only_camera2_correct" in confident_data:
                    pose_data["only_camera2_correct"] = confident_data["only_camera2_correct"]
                else:
                    pose_data["only_camera2_correct"] = []
                
                if "only_camera2_wrong" in confident_data:
                    pose_data["only_camera2_wrong"] = confident_data["only_camera2_wrong"]
                else:
                    pose_data["only_camera2_wrong"] = []
                    
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in {confident_file}")
                pose_data["only_camera2_correct"] = []
                pose_data["only_camera2_wrong"] = []
        else:
            # No confident data available
            pose_data["only_camera2_correct"] = []
            pose_data["only_camera2_wrong"] = []
        
        # Save merged data with custom formatting
        output_file = Path(output_folder) / pose_file.name
        save_pose_json(pose_data, output_file)
        
        processed_count += 1
        print(f"Processed: {pose_file.name}")
    
    print(f"\n✓ Successfully processed {processed_count} files")
    print(f"Output saved to: {output_folder}")

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
            f.write(f'  "{camera}": {{\n')
            camera_data = data[camera]
            joints = list(camera_data.keys())
            
            for j, joint in enumerate(joints):
                coords = camera_data[joint]
                coords_str = json.dumps(coords)
                
                if j < len(joints) - 1:
                    f.write(f'    "{joint}": {coords_str},\n')
                else:
                    f.write(f'    "{joint}": {coords_str}\n')
            
            if i < len(cameras) - 1 or metadata_keys:
                f.write('  },\n')
            else:
                f.write('  }\n')
        
        # Write metadata (only_camera2_correct, only_camera2_wrong)
        for i, key in enumerate(metadata_keys):
            value_str = json.dumps(data[key])
            
            if i < len(metadata_keys) - 1:
                f.write(f'  "{key}": {value_str},\n')
            else:
                f.write(f'  "{key}": {value_str}\n')
        
        f.write('}\n')

if __name__ == "__main__":
    # Configuration
    pose_folder = "output/merged_poses"  # Folder containing pose JSON files
    confident_folder = "output/confident"  # Folder containing confident JSON files
    output_folder = "output/merged_total"  # Output folder
    
    process_pose_with_confident(pose_folder, confident_folder, output_folder)
