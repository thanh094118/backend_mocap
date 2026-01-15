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
        
        # Bước 1: Extract images
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 1,
            'total_steps': 7, 
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
            raise Exception(f"Image extraction failed: {result.stderr}")
        
        logger.info(f"Step 1 completed. Output: {result.stdout}")
        
        # Bước 2a: EMC for camera1
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 2,
            'total_steps': 7,
            'message': 'Processing camera1 (pose estimation)...'
        }
        logger.info(f"Step 2a: Processing camera1 for job {job_id}")
        
        camera1_output = Path(work_dir) / 'output' / 'camera1'
        camera1_output.mkdir(parents=True, exist_ok=True)
        
        # Auto-answer prompts with 'y\ny\nn\n'
        emc_input = "y\ny\nn\n"
        result = subprocess.run(
            ['emc', '--data', '/app/config/datasets/svimage.yml', 
             '--exp', '/app/config/1v1p/hrnet_pare_finetune.yml', 
             '--root', str(camera1_output), 
             '--subs', 'images'],
            input=emc_input,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=1800  # 30 minutes
        )
        if result.returncode != 0:
            raise Exception(f"Camera1 processing failed: {result.stderr}")
        
        logger.info(f"Step 2a completed")
        
        # Bước 2b: EMC for camera2
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 2.5,
            'total_steps': 7,
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
            input=emc_input,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=1800
        )
        if result.returncode != 0:
            raise Exception(f"Camera2 processing failed: {result.stderr}")
        
        logger.info(f"Step 2b completed")
        
        # Bước 3: Merge poses
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 3,
            'total_steps': 7,
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
            raise Exception(f"Pose merging failed: {result.stderr}")
        
        logger.info(f"Step 3 completed")
        
        # Bước 4: Pixel processing
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 4,
            'total_steps': 7,
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
            raise Exception(f"Pixel processing failed: {result.stderr}")
        
        logger.info(f"Step 4 completed")
        
        # Bước 5: Total merge
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 5,
            'total_steps': 7,
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
            raise Exception(f"Total merge failed: {result.stderr}")
        
        logger.info(f"Step 5 completed")
        
        # Bước 6: Alignment
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 6,
            'total_steps': 7,
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
            raise Exception(f"Alignment failed: {result.stderr}")
        
        logger.info(f"Step 6 completed")
        
        # Bước 7: Zip output
        processing_status[job_id] = {
            'status': 'processing', 
            'step': 7,
            'total_steps': 7,
            'message': 'Packaging results...'
        }
        logger.info(f"Step 7: Zipping results for job {job_id}")
        
        output_folder = Path(work_dir) / 'output' / 'output_final'
        
        if not output_folder.exists():
            raise Exception(f"Output folder not found: {output_folder}")
        
        # Check if there are files in output
        output_files = list(output_folder.rglob('*'))
        if not output_files:
            raise Exception("No output files generated")
        
        logger.info(f"Found {len(output_files)} files to zip")
        
        zip_path = Path(work_dir) / 'results.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(output_folder)
                    zipf.write(file_path, arcname)
                    logger.info(f"Added to zip: {arcname}")
        
        processing_status[job_id] = {
            'status': 'completed', 
            'step': 7,
            'total_steps': 7,
            'message': 'Processing completed successfully',
            'zip_path': str(zip_path),
            'output_files': len(output_files)
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
    
    Request:
    - video1: MP4 file (camera 1)
    - video2: MP4 file (camera 2)
    
    Response:
    - job_id: ID để track tiến trình
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
        
        # Tạo thư mục làm việc
        work_dir = Path('/app/temp') / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Store work directory
        job_directories[job_id] = str(work_dir)
        
        # Lưu videos
        video1_path = work_dir / 'camera1.mp4'
        video2_path = work_dir / 'camera2.mp4'
        
        video1.save(str(video1_path))
        video2.save(str(video2_path))
        
        logger.info(f"Videos saved for job {job_id}")
        logger.info(f"Video1: {video1_path} ({video1_path.stat().st_size / 1024 / 1024:.2f} MB)")
        logger.info(f"Video2: {video2_path} ({video2_path.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Initialize status
        processing_status[job_id] = {
            'status': 'queued',
            'step': 0,
            'total_steps': 7,
            'message': 'Job queued'
        }
        
        # Khởi động pipeline trong background thread
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
        'total_steps': status.get('total_steps', 7),
        'message': status.get('message', ''),
        'progress': f"{status.get('step', 0)}/{status.get('total_steps', 7)}"
    }
    
    if status['status'] == 'completed':
        response['download_url'] = f'/api/download/{job_id}'
        response['output_files'] = status.get('output_files', 0)
    
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
