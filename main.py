# ==============================================================================
# 🔍 DIAGNOSTIC SCRIPT: KIỂM TRA FILE HỆ THỐNG
# ==============================================================================
import os
import sys
from pathlib import Path

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def scan_models():
    print_separator("BẮT ĐẦU QUÉT FILE MODEL TRONG /app")
    
    # 1. In thư mục hiện tại
    cwd = os.getcwd()
    print(f"📍 Current Working Directory: {cwd}")
    
    # 2. Định nghĩa các đuôi file cần tìm
    extensions = {'.pt', '.pth', '.ckpt', '.pkl', '.h5', '.json'}
    found_count = 0
    
    # 3. Quét đệ quy từ thư mục gốc /app
    # Nếu bạn chạy local không có /app thì thay bằng '.'
    search_root = '/app' if os.path.exists('/app') else '.'
    
    print(f"🚀 Scanning root: {search_root} ...\n")

    for root, dirs, files in os.walk(search_root):
        for file in files:
            # Lấy đuôi file
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                full_path = os.path.join(root, file)
                try:
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    print(f"  ✅ FOUND: {full_path}")
                    print(f"     └─ Size: {size_mb:.2f} MB")
                    found_count += 1
                except Exception as e:
                    print(f"  ⚠️  FOUND BUT ERROR: {full_path} ({str(e)})")

    if found_count == 0:
        print("\n❌ CẢNH BÁO: KHÔNG TÌM THẤY BẤT KỲ FILE MODEL NÀO!")
    else:
        print(f"\n✨ Tổng cộng tìm thấy: {found_count} file models.")

    # 4. Kiểm tra cụ thể file đang bị lỗi của bạn
    print_separator("KIỂM TRA CỤ THỂ FILE YOLO")
    target_yolo = "/app/models/yolo/yolov5m.pt"
    if os.path.exists(target_yolo):
        sz = os.path.getsize(target_yolo) / (1024 * 1024)
        print(f"🎯 YOLO STATUS: [OK] File tồn tại tại {target_yolo} ({sz:.2f} MB)")
    else:
        print(f"🎯 YOLO STATUS: [MISSING] Không thấy file tại {target_yolo}")
        # Gợi ý fix
        print("   👉 Gợi ý: Kiểm tra xem file có bị lưu nhầm vào /app/data/models/... không?")

    print("="*60 + "\n")

# Chạy ngay lập tức
scan_models()
# ==============================================================================

# ... Code import Flask và các phần còn lại của bạn ở dưới này ...
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import tempfile
import shutil
import os
import subprocess
import zipfile
from pathlib import Path
import threading
import uuid
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Store processing status and temp directories
processing_status = {}
job_directories = {}

def log_directory_structure(startpath):
    """In ra cấu trúc thư mục để debug"""
    logger.info(f"--- Scanning directory: {startpath} ---")
    try:
        for root, dirs, files in os.walk(startpath):
            level = root.replace(str(startpath), '').count(os.sep)
            indent = ' ' * 4 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            # Chỉ in 5 file đầu tiên để log đỡ dài
            for f in files[:5]:
                logger.info(f"{subindent}{f}")
            if len(files) > 5:
                logger.info(f"{subindent}... ({len(files)} total files)")
    except Exception as e:
        logger.error(f"Error scanning directory: {e}")
    logger.info("---------------------------------------")

def allowed_file(filename):
    return filename.lower().endswith('.mp4')

def cleanup_job(job_id):
    """Cleanup temporary directory after job completion"""
    if job_id in job_directories:
        temp_dir = job_directories[job_id]
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            del job_directories[job_id]
            logger.info(f"Cleaned up job {job_id}")
        except Exception as e:
            logger.error(f"Error cleaning up job {job_id}: {e}")

def run_pipeline(job_id, video1_path, video2_path, work_dir):
    """Chạy toàn bộ pipeline xử lý"""
    try:
        logger.info(f"Starting pipeline for job {job_id}")
        logger.info(f"Working directory: {work_dir}")
        
        # Bước 0: Download models (nếu chưa có)
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 0, 
            'total_steps': 6,
            'message': 'Checking models...'
        }
        
        # Bước 1: Extract images
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 1,
            'total_steps': 6, 
            'message': 'Extracting images from videos...'
        }
        logger.info(f"Step 1: Extracting images for job {job_id}")
        
        result = subprocess.run(
            ['python', '/app/myscript/extract_image.py', video1_path, video2_path],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=300
        )
        if result.returncode != 0:
            logger.error(f"Extract image stderr: {result.stderr}")
            logger.error(f"Extract image stdout: {result.stdout}")
            raise Exception(f"Image extraction failed: {result.stderr}")
        
        logger.info(f"Step 1 completed. Output: {result.stdout}")
        
        # Debug: Check what was created
        log_directory_structure(work_dir)
        
        # Bước 2a: EMC for camera1
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 2,
            'total_steps': 6,
            'message': 'Processing camera1 (pose estimation)...'
        }
        logger.info(f"Step 2a: Processing camera1 for job {job_id}")
        
        camera1_output = Path(work_dir) / 'output' / 'camera1'
        camera1_output.mkdir(parents=True, exist_ok=True)
        
        # Run EMC without auto-answer prompts
        result = subprocess.run(
            ['emc', '--data', '/app/config/datasets/svimage.yml', 
             '--exp', '/app/config/1v1p/hrnet_pare_finetune.yml', 
             '--root', str(camera1_output), 
             '--subs', 'images'],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=1800  # 30 minutes
        )
        if result.returncode != 0:
            logger.error(f"Camera1 EMC stderr: {result.stderr}")
            logger.error(f"Camera1 EMC stdout: {result.stdout}")
            raise Exception(f"Camera1 processing failed: {result.stderr}")
        
        logger.info(f"Step 2a completed")
        log_directory_structure(camera1_output)
        
        # Bước 2b: EMC for camera2
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 2.5,
            'total_steps': 6,
            'message': 'Processing camera2 (pose estimation)...'
        }
        logger.info(f"Step 2b: Processing camera2 for job {job_id}")
        
        camera2_output = Path(work_dir) / 'output' / 'camera2'
        camera2_output.mkdir(parents=True, exist_ok=True)
        
        result = subprocess.run(
            ['emc', '--data', '/app/config/datasets/svimage.yml', 
             '--exp', '/app/config/1v1p/hrnet_pare_finetune.yml', 
             '--root', str(camera2_output), 
             '--subs', 'images'],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=1800
        )
        if result.returncode != 0:
            logger.error(f"Camera2 EMC stderr: {result.stderr}")
            logger.error(f"Camera2 EMC stdout: {result.stdout}")
            raise Exception(f"Camera2 processing failed: {result.stderr}")
        
        logger.info(f"Step 2b completed")
        log_directory_structure(camera2_output)
        
        # Bước 3: Merge poses
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 3,
            'total_steps': 6,
            'message': 'Merging poses from both cameras...'
        }
        logger.info(f"Step 3: Merging poses for job {job_id}")
        
        result = subprocess.run(
            ['python', '/app/myscript/merged_poses.py'],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=300
        )
        if result.returncode != 0:
            logger.error(f"Merged poses stderr: {result.stderr}")
            logger.error(f"Merged poses stdout: {result.stdout}")
            raise Exception(f"Pose merging failed: {result.stderr}")
        
        logger.info(f"Step 3 completed. Output: {result.stdout}")
        
        # Bước 4: Pixel processing
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 4,
            'total_steps': 6,
            'message': 'Processing pixel data...'
        }
        logger.info(f"Step 4: Pixel processing for job {job_id}")
        
        result = subprocess.run(
            ['python', '/app/myscript/pixel.py'],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=300
        )
        if result.returncode != 0:
            logger.error(f"Pixel processing stderr: {result.stderr}")
            logger.error(f"Pixel processing stdout: {result.stdout}")
            raise Exception(f"Pixel processing failed: {result.stderr}")
        
        logger.info(f"Step 4 completed. Output: {result.stdout}")
        
        # Bước 5: Total merge
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 5,
            'total_steps': 6,
            'message': 'Performing final merge...'
        }
        logger.info(f"Step 5: Total merge for job {job_id}")
        
        result = subprocess.run(
            ['python', '/app/myscript/merged_total.py'],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=300
        )
        if result.returncode != 0:
            logger.error(f"Total merge stderr: {result.stderr}")
            logger.error(f"Total merge stdout: {result.stdout}")
            raise Exception(f"Total merge failed: {result.stderr}")
        
        logger.info(f"Step 5 completed. Output: {result.stdout}")
        
        # Bước 6: Alignment
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 6,
            'total_steps': 6,
            'message': 'Aligning final output...'
        }
        logger.info(f"Step 6: Alignment for job {job_id}")
        
        result = subprocess.run(
            ['python', '/app/myscript/aligned.py'],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=300
        )
        if result.returncode != 0:
            logger.error(f"Alignment stderr: {result.stderr}")
            logger.error(f"Alignment stdout: {result.stdout}")
            raise Exception(f"Alignment failed: {result.stderr}")
        
        logger.info(f"Step 6 completed. Output: {result.stdout}")
        
        # Check output folder
        output_folder = Path(work_dir) / 'output' / 'output_final'
        logger.info(f"Looking for output at: {output_folder}")
        
        if not output_folder.exists():
            logger.error(f"Output folder not found: {output_folder}")
            # Log what exists in output directory
            output_dir = Path(work_dir) / 'output'
            if output_dir.exists():
                log_directory_structure(output_dir)
            raise Exception(f"Output folder not found: {output_folder}")
        
        # Debug: Show output structure
        log_directory_structure(output_folder)
        
        # Check if there are files in output
        output_files = list(output_folder.rglob('*'))
        output_files = [f for f in output_files if f.is_file()]  # Only files, not directories
        
        if not output_files:
            raise Exception("No output files generated")
        
        logger.info(f"Found {len(output_files)} files to zip")
        for f in output_files[:10]:  # Log first 10 files
            logger.info(f"  - {f.relative_to(output_folder)}")
        
        # Zip output
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 6,
            'total_steps': 6,
            'message': 'Packaging results...'
        }
        logger.info(f"Creating zip file for job {job_id}")
        
        zip_path = Path(work_dir) / 'results.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_files:
                arcname = file_path.relative_to(output_folder)
                zipf.write(file_path, arcname)
                logger.info(f"Added to zip: {arcname}")
        
        zip_size = zip_path.stat().st_size / 1024 / 1024  # Size in MB
        logger.info(f"Zip file created: {zip_path} ({zip_size:.2f} MB)")
        
        processing_status[job_id] = {
            'status': 'completed', 
            'step': 6,
            'total_steps': 6,
            'message': 'Processing completed successfully',
            'zip_path': str(zip_path),
            'output_files': len(output_files),
            'zip_size_mb': round(zip_size, 2)
        }
        
        logger.info(f"Pipeline completed for job {job_id}")
        
    except subprocess.TimeoutExpired as e:
        error_msg = f"Step timeout: {str(e)}"
        logger.error(f"Job {job_id} failed: {error_msg}")
        processing_status[job_id] = {
            'status': 'failed',
            'message': error_msg
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Job {job_id} failed: {error_msg}")
        processing_status[job_id] = {
            'status': 'failed',
            'message': error_msg
        }

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running',
        'service': 'Video Processing Pipeline',
        'version': '1.0',
        'endpoints': {
            'POST /api/upload': 'Upload 2 videos and start processing',
            'GET /api/status/<job_id>': 'Check processing status',
            'GET /api/download/<job_id>': 'Download results',
            'GET /api/jobs': 'List all jobs',
            'DELETE /api/job/<job_id>': 'Cancel and cleanup job'
        }
    })

@app.route('/api/upload', methods=['POST'])
def upload_videos():
    """
    Upload 2 video files và bắt đầu xử lý
    """
    try:
        # Validate files
        if 'video1' not in request.files or 'video2' not in request.files:
            return jsonify({'error': 'Both video1 and video2 are required'}), 400
        
        video1 = request.files['video1']
        video2 = request.files['video2']
        
        if video1.filename == '' or video2.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(video1.filename) or not allowed_file(video2.filename):
            return jsonify({'error': 'Only MP4 files are allowed'}), 400
        
        # Tạo job ID
        job_id = str(uuid.uuid4())
        
        # Tạo thư mục làm việc gốc (/app/temp/<job_id>)
        work_dir = Path('/app/temp') / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # --- [START FIX] TẠO THÊM FOLDER INPUT ---
        # Tạo folder con 'input' để script extract_image.py tìm thấy
        input_dir = work_dir / 'input'
        input_dir.mkdir(exist_ok=True)
        
        # Cập nhật đường dẫn lưu file vào trong folder input
        video1_path = input_dir / 'camera1.mp4'
        video2_path = input_dir / 'camera2.mp4'
        # --- [END FIX] ---

        # Store work directory
        job_directories[job_id] = str(work_dir)
        
        # Lưu videos
        video1.save(str(video1_path))
        video2.save(str(video2_path))
        
        logger.info(f"Videos saved for job {job_id}")
        logger.info(f"Video1: {video1_path} ({video1_path.stat().st_size / 1024 / 1024:.2f} MB)")
        logger.info(f"Video2: {video2_path} ({video2_path.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Initialize status
        processing_status[job_id] = {
            'status': 'queued',
            'step': 0,
            'total_steps': 7, # Lưu ý: Nên để tổng số bước khớp với logic (7 bước)
            'message': 'Job queued'
        }
        
        # Khởi động pipeline
        # Lưu ý: video1_path giờ đã là đường dẫn nằm trong folder input
        # thread sẽ truyền đúng đường dẫn mới này cho hàm run_pipeline
        thread = threading.Thread(
            target=run_pipeline,
            args=(job_id, str(video1_path), str(video2_path), str(work_dir)),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'Processing started',
            'status': 'queued',
            'status_url': f'/api/status/{job_id}',
            'download_url': f'/api/download/{job_id}'
        }), 202
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Kiểm tra trạng thái xử lý"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = processing_status[job_id]
    response = {
        'job_id': job_id,
        'status': status['status'],
        'step': status.get('step', 0),
        'total_steps': status.get('total_steps', 6),
        'message': status.get('message', ''),
        'progress': f"{status.get('step', 0)}/{status.get('total_steps', 6)}"
    }
    
    if status['status'] == 'completed':
        response['download_url'] = f'/api/download/{job_id}'
        response['output_files'] = status.get('output_files', 0)
        response['zip_size_mb'] = status.get('zip_size_mb', 0)
    
    return jsonify(response)

@app.route('/api/download/<job_id>', methods=['GET'])
def download_results(job_id):
    """Download kết quả đã xử lý"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = processing_status[job_id]
    
    if status['status'] != 'completed':
        return jsonify({
            'error': 'Processing not completed yet',
            'current_status': status['status'],
            'message': status.get('message', '')
        }), 400
    
    zip_path = status.get('zip_path')
    
    if not zip_path or not os.path.exists(zip_path):
        return jsonify({'error': 'Result file not found'}), 404
    
    return send_file(
        zip_path,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'results_{job_id}.zip'
    )

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    jobs = []
    for job_id, status in processing_status.items():
        jobs.append({
            'job_id': job_id,
            'status': status['status'],
            'step': status.get('step', 0),
            'message': status.get('message', '')
        })
    
    return jsonify({
        'total': len(jobs),
        'jobs': jobs
    })

@app.route('/api/job/<job_id>', methods=['DELETE'])
def cancel_job(job_id):
    """Cancel and cleanup a job"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    cleanup_job(job_id)
    
    if job_id in processing_status:
        del processing_status[job_id]
    
    return jsonify({
        'message': f'Job {job_id} cancelled and cleaned up'
    })

if __name__ == '__main__':
    # Ensure temp directory exists
    Path('/app/temp').mkdir(parents=True, exist_ok=True)
    Path('/app/output').mkdir(parents=True, exist_ok=True)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
