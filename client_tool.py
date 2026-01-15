"""
==============================================
FILE 3: client_tool.py - Script cho User
==============================================
"""
import requests
import sys
import time
from pathlib import Path

SERVER_URL = "https://your-app.onrender.com"  # Thay bằng URL Render của bạn

def upload_videos(video1_path, video2_path):
    """Upload 2 videos lên server"""
    print("Uploading videos...")
    
    files = {
        'video1': open(video1_path, 'rb'),
        'video2': open(video2_path, 'rb')
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/api/upload", files=files)
        
        if response.status_code == 202:
            data = response.json()
            print(f"✓ Upload successful!")
            print(f"Job ID: {data['job_id']}")
            return data['job_id']
        else:
            print(f"✗ Upload failed: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    finally:
        files['video1'].close()
        files['video2'].close()

def check_status(job_id):
    """Kiểm tra trạng thái xử lý"""
    try:
        response = requests.get(f"{SERVER_URL}/api/status/{job_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"✗ Status check failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def download_results(job_id, output_path='results.zip'):
    """Download kết quả"""
    print("Downloading results...")
    
    try:
        response = requests.get(f"{SERVER_URL}/api/download/{job_id}", stream=True)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Results saved to: {output_path}")
            return True
        else:
            print(f"✗ Download failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Main workflow"""
    if len(sys.argv) != 3:
        print("Usage: python client_tool.py <video1.mp4> <video2.mp4>")
        sys.exit(1)
    
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    
    # Validate files exist
    if not Path(video1_path).exists():
        print(f"Error: {video1_path} not found")
        sys.exit(1)
    
    if not Path(video2_path).exists():
        print(f"Error: {video2_path} not found")
        sys.exit(1)
    
    print("="*60)
    print("Video Processing Client")
    print("="*60)
    
    # Step 1: Upload
    job_id = upload_videos(video1_path, video2_path)
    
    if not job_id:
        print("Failed to start processing")
        sys.exit(1)
    
    # Step 2: Monitor progress
    print("\nMonitoring progress...")
    
    while True:
        status = check_status(job_id)
        
        if not status:
            time.sleep(5)
            continue
        
        current_status = status['status']
        step = status.get('step', 0)
        message = status.get('message', '')
        
        print(f"\rStep {step}/7: {message}", end='', flush=True)
        
        if current_status == 'completed':
            print("\n\n✓ Processing completed!")
            break
        elif current_status == 'failed':
            print(f"\n\n✗ Processing failed: {message}")
            sys.exit(1)
        
        time.sleep(5)  # Check mỗi 5 giây
    
    # Step 3: Download results
    print()
    if download_results(job_id):
        print("\n" + "="*60)
        print("✓ All done! Check results.zip")
        print("="*60)
    else:
        print("\n✗ Failed to download results")
        sys.exit(1)

if __name__ == "__main__":
    main()


