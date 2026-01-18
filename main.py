from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import subprocess
import zipfile
from pathlib import Path
import uuid
import logging
import shutil
from datetime import datetime
import threading
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Single worker - chỉ xử lý 1 job tại 1 thời điểm
current_job = {
    'job_id': None,
    'status': 'idle',  # idle, processing, completed, failed
    'stage': '',
    'result_path': None,
    'error': None,
    'started_at': None,
    'completed_at': None
}
job_lock = threading.Lock()

TEMP_DIR = Path('/app/temp')
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def cleanup_old_jobs():
    """Tự động xóa các job cũ hơn 1 giờ"""
    try:
        cutoff_time = time.time() - 3600  # 1 hour
        for job_dir in TEMP_DIR.iterdir():
            if job_dir.is_dir():
                dir_mtime = job_dir.stat().st_mtime
                if dir_mtime < cutoff_time:
                    logger.info(f"🗑️  Auto cleanup: Removing old job directory: {job_dir.name}")
                    shutil.rmtree(job_dir)
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")

# def cleanup_job_directory(job_id):
#     """Xóa thư mục của 1 job cụ thể"""
#     job_dir = TEMP_DIR / job_id
#     if job_dir.exists():
#         try:
#             shutil.rmtree(job_dir)
#             logger.info(f"🗑️  Cleaned up job directory: {job_id}")
#         except Exception as e:
#             logger.error(f"❌ Failed to cleanup {job_id}: {e}")

def run_pipeline(job_id, video1_path, video2_path, work_dir):
    """Pipeline xử lý video"""
    global current_job
    
    work_dir = Path(work_dir)
    
    try:
        logger.info(f"{'='*60}")
        logger.info(f"🚀 STARTING JOB: {job_id}")
        logger.info(f"{'='*60}")
        
        # Stage 1: Extract images
        current_job['stage'] = 'Stage 1/6: Extracting images'
        logger.info(f"📍 {current_job['stage']}")
        logger.info(f"   Input: {video1_path}, {video2_path}")
        logger.info(f"   Output: {work_dir}/images/")
        
        result = subprocess.run(
            ['python', '/app/myscript/extract_image.py', video1_path, video2_path],
            capture_output=True, text=True, cwd=work_dir, timeout=300
        )
        if result.returncode != 0:
            raise Exception(f"Extract failed: {result.stderr}")
        logger.info(f"   ✅ Stage 1 completed")
        
        # Stage 2: Process camera1
        current_job['stage'] = 'Stage 2/6: Processing camera1'
        logger.info(f"📍 {current_job['stage']}")
        camera1_output = work_dir / 'output' / 'camera1'
        camera1_output.mkdir(parents=True, exist_ok=True)
        logger.info(f"   Input: {work_dir}/images/camera1/")
        logger.info(f"   Output: {camera1_output}")

        result = subprocess.run(
            [
                'emc', 
                '--data', '/app/config/datasets/svimage.yml', 
                '--exp', '/app/config/1v1p/hrnet_pare_finetune.yml', 
                '--root', str(camera1_output), 
                '--subs', 'images',
            ],
            capture_output=True, text=True, cwd=work_dir, timeout=3600
        )
        if result.returncode != 0:
            raise Exception(f"Camera1 failed: {result.stderr}")
        logger.info(f"   ✅ Stage 2 completed")
        
        # Stage 3: Process camera2
        current_job['stage'] = 'Stage 3/6: Processing camera2'
        logger.info(f"📍 {current_job['stage']}")
        camera2_output = work_dir / 'output' / 'camera2'
        camera2_output.mkdir(parents=True, exist_ok=True)
        logger.info(f"   Input: {work_dir}/images/camera2/")
        logger.info(f"   Output: {camera2_output}")
        
        result = subprocess.run(
            [
                'emc', 
                '--data', '/app/config/datasets/svimage.yml', 
                '--exp', '/app/config/1v1p/hrnet_pare_finetune.yml', 
                '--root', str(camera1_output), 
                '--subs', 'images',
            ],
            capture_output=True, text=True, cwd=work_dir, timeout=3600
        )
        if result.returncode != 0:
            raise Exception(f"Camera2 failed: {result.stderr}")
        logger.info(f"   ✅ Stage 3 completed")
        
        # Stage 4: Merge poses
        current_job['stage'] = 'Stage 4/6: Merging poses'
        logger.info(f"📍 {current_job['stage']}")
        
        # 1. Kiểm tra xem file đầu vào có tồn tại không trước khi chạy
        check_path_1 = work_dir / 'output' / 'camera1'
        check_path_2 = work_dir / 'output' / 'camera2'
        
        # Đếm file để debug
        try:
            files_1 = list(check_path_1.rglob('*'))
            files_2 = list(check_path_2.rglob('*'))
            logger.info(f"🔎 DEBUG: Camera1 files: {len(files_1)}, Camera2 files: {len(files_2)}")
        except Exception as e:
            logger.error(f"🔎 DEBUG Error check files: {e}")

        logger.info(f"   Input: {check_path_1}, {check_path_2}")
        logger.info(f"   Output: {work_dir}/output/merged_poses/")
        
        # 2. Chạy lệnh
        result = subprocess.run(
            ['python', '/app/myscript/merged_poses.py'],
            capture_output=True, text=True, cwd=work_dir, timeout=600
        )
        
        # 3. In lỗi chi tiết nếu thất bại
        if result.returncode != 0:
            logger.error(f"❌ STDERR (Lỗi chi tiết): {result.stderr}")
            logger.error(f"❌ STDOUT (Log chạy): {result.stdout}")
            raise Exception(f"Merge poses failed: {result.stderr}")
            
        logger.info(f"   ✅ Stage 4 completed")
        
        # Stage 5: Pixel processing
        current_job['stage'] = 'Stage 5/6: Pixel processing'
        logger.info(f"📍 {current_job['stage']}")
        logger.info(f"   Input: {work_dir}/output/merged_poses/")
        logger.info(f"   Output: {work_dir}/output/pixels/")
        
        result = subprocess.run(
            ['python', '/app/myscript/pixel.py'],
            capture_output=True, text=True, cwd=work_dir, timeout=300
        )
        if result.returncode != 0:
            raise Exception(f"Pixel processing failed: {result.stderr}")
        logger.info(f"   ✅ Stage 5 completed")
        
        # Stage 6: Final merge and alignment
        current_job['stage'] = 'Stage 6/6: Final processing'
        logger.info(f"📍 {current_job['stage']}")
        logger.info(f"   Input: {work_dir}/output/pixels/")
        
        result = subprocess.run(
            ['python', '/app/myscript/merged_total.py'],
            capture_output=True, text=True, cwd=work_dir, timeout=300
        )
        if result.returncode != 0:
            raise Exception(f"Total merge failed: {result.stderr}")
        
        result = subprocess.run(
            ['python', '/app/myscript/aligned.py'],
            capture_output=True, text=True, cwd=work_dir, timeout=300
        )
        if result.returncode != 0:
            raise Exception(f"Alignment failed: {result.stderr}")
        logger.info(f"   ✅ Stage 6 completed")
        
        # Create zip file
        logger.info(f"📦 Creating result package...")
        output_folder = work_dir / 'output' / 'output_final'
        logger.info(f"   Input: {output_folder}")
        
        if not output_folder.exists():
            raise Exception(f"Output folder not found: {output_folder}")
        
        output_files = [f for f in output_folder.rglob('*') if f.is_file()]
        if not output_files:
            raise Exception("No output files generated")
        
        zip_path = work_dir / 'results.zip'
        logger.info(f"   Output: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_files:
                arcname = file_path.relative_to(output_folder)
                zipf.write(file_path, arcname)
        
        zip_size = zip_path.stat().st_size / 1024 / 1024
        logger.info(f"   ✅ Zip created: {zip_size:.2f} MB, {len(output_files)} files")
        
        # Update status
        with job_lock:
            current_job['status'] = 'completed'
            current_job['stage'] = 'Completed'
            current_job['result_path'] = str(zip_path)
            current_job['completed_at'] = datetime.now().isoformat()
        
        logger.info(f"{'='*60}")
        logger.info(f"✅ JOB COMPLETED: {job_id}")
        logger.info(f"{'='*60}")
        
    except subprocess.TimeoutExpired as e:
        error_msg = f"Timeout at {current_job['stage']}: {str(e)}"
        logger.error(f"❌ {error_msg}")
        with job_lock:
            current_job['status'] = 'failed'
            current_job['error'] = error_msg
            current_job['completed_at'] = datetime.now().isoformat()
        cleanup_job_directory(job_id)
        
    except Exception as e:
        error_msg = f"Error at {current_job['stage']}: {str(e)}"
        logger.error(f"❌ {error_msg}")
        with job_lock:
            current_job['status'] = 'failed'
            current_job['error'] = error_msg
            current_job['completed_at'] = datetime.now().isoformat()
        cleanup_job_directory(job_id)

@app.route('/api/process', methods=['POST'])
def process_videos():
    """
    Upload 2 videos và xử lý
    - Chỉ nhận 1 job tại 1 thời điểm
    - Tự động cleanup job cũ khi hoàn thành
    """
    global current_job
    
    # Check if busy
    with job_lock:
        if current_job['status'] == 'processing':
            elapsed = None
            if current_job['started_at']:
                start_time = datetime.fromisoformat(current_job['started_at'])
                elapsed_seconds = (datetime.now() - start_time).total_seconds()
                elapsed = f"{int(elapsed_seconds // 60)}m {int(elapsed_seconds % 60)}s"
            
            return jsonify({
                'error': 'Server đang xử lý 1 job khác',
                'message': 'Hệ thống chỉ xử lý 1 video tại 1 thời điểm. Vui lòng đợi job hiện tại hoàn thành.',
                'current_job': {
                    'job_id': current_job['job_id'],
                    'stage': current_job['stage'],
                    'started_at': current_job['started_at'],
                    'elapsed_time': elapsed
                },
                'suggestion': f'Kiểm tra tiến độ tại: GET /api/status'
            }), 503
    
    try:
        # Validate files
        if 'video1' not in request.files or 'video2' not in request.files:
            return jsonify({'error': 'Both video1 and video2 are required'}), 400
        
        video1 = request.files['video1']
        video2 = request.files['video2']
        
        if not video1.filename or not video2.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        if not (video1.filename.lower().endswith('.mp4') and 
                video2.filename.lower().endswith('.mp4')):
            return jsonify({'error': 'Only MP4 files are allowed'}), 400
        
        # Cleanup old jobs trước khi bắt đầu job mới
        cleanup_old_jobs()
        
        # Create new job
        job_id = str(uuid.uuid4())
        work_dir = TEMP_DIR / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        
        input_dir = work_dir / 'input'
        input_dir.mkdir(exist_ok=True)
        
        video1_path = input_dir / 'camera1.mp4'
        video2_path = input_dir / 'camera2.mp4'
        
        # Save videos
        video1.save(str(video1_path))
        video2.save(str(video2_path))
        
        v1_size = video1_path.stat().st_size / 1024 / 1024
        v2_size = video2_path.stat().st_size / 1024 / 1024
        logger.info(f"📥 Videos uploaded: {v1_size:.2f}MB, {v2_size:.2f}MB")
        
        # Update status
        with job_lock:
            current_job = {
                'job_id': job_id,
                'status': 'processing',
                'stage': 'Starting...',
                'result_path': None,
                'error': None,
                'started_at': datetime.now().isoformat(),
                'completed_at': None
            }
        
        # Start processing
        thread = threading.Thread(
            target=run_pipeline,
            args=(job_id, str(video1_path), str(video2_path), str(work_dir)),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'message': 'Đã bắt đầu xử lý video.',
            'next_step': f'Kiểm tra tiến độ tại: GET /api/status',
            'result_url': f'/api/result/{job_id}'
        }), 202
        
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """
    Kiểm tra trạng thái và download kết quả
    - Nếu đang xử lý: trả về status
    - Nếu hoàn thành: tự động download file và cleanup
    - Nếu lỗi: trả về error
    """
    global current_job
    
    with job_lock:
        # Check if this is the current job
        if current_job['job_id'] != job_id:
            return jsonify({
                'error': 'Không tìm thấy job',
                'message': f'Job {job_id} không tồn tại hoặc đã bị xóa.',
                'current_job_id': current_job['job_id'] if current_job['status'] != 'idle' else None
            }), 404
        
        status = current_job['status']
        
        # Still processing
        if status == 'processing':
            elapsed_time = None
            if current_job['started_at']:
                start_time = datetime.fromisoformat(current_job['started_at'])
                elapsed_seconds = (datetime.now() - start_time).total_seconds()
                elapsed_minutes = int(elapsed_seconds // 60)
                elapsed_secs = int(elapsed_seconds % 60)
                elapsed_time = f"{elapsed_minutes}m {elapsed_secs}s"
            
            return jsonify({
                'job_id': job_id,
                'status': 'processing',
                'stage': current_job['stage'],
                'started_at': current_job['started_at'],
                'elapsed_time': elapsed_time,
                'message': 'Job đang được xử lý. Vui lòng kiểm tra lại sau.',
                'check_status': '/api/status'
            }), 202
        
        # Failed
        if status == 'failed':
            error = current_job['error']
            # Reset to idle sau khi trả lỗi
            current_job = {
                'job_id': None,
                'status': 'idle',
                'stage': '',
                'result_path': None,
                'error': None,
                'started_at': None,
                'completed_at': None
            }
            return jsonify({
                'job_id': job_id,
                'status': 'failed',
                'error': error,
                'completed_at': current_job.get('completed_at')
            }), 500
        
        # Completed - download file
        if status == 'completed':
            result_path = current_job['result_path']
            
            if not result_path or not os.path.exists(result_path):
                return jsonify({'error': 'Result file not found'}), 404
            
            logger.info(f"📤 Sending result: {job_id}")
            
            # Send file trước
            response = send_file(
                result_path,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'results_{job_id}.zip'
            )
            
            # Cleanup sau khi download (sẽ xảy ra sau khi response được gửi)
            @response.call_on_close
            def cleanup():
                cleanup_job_directory(job_id)
                with job_lock:
                    current_job['job_id'] = None
                    current_job['status'] = 'idle'
                    current_job['stage'] = ''
                    current_job['result_path'] = None
                    current_job['started_at'] = None
                    current_job['completed_at'] = None
            
            return response

@app.route('/api/status', methods=['GET'])
def server_status():
    """Kiểm tra trạng thái server và tiến độ xử lý chi tiết"""
    with job_lock:
        if current_job['status'] == 'idle':
            return jsonify({
                'server_status': 'Sẵn sàng',
                'message': 'Server đang rảnh, có thể nhận job mới',
                'current_job': None
            })
        
        # Tính thời gian đã chạy
        elapsed_time = None
        estimated_remaining = None
        if current_job['started_at']:
            start_time = datetime.fromisoformat(current_job['started_at'])
            elapsed_seconds = (datetime.now() - start_time).total_seconds()
            elapsed_minutes = int(elapsed_seconds // 60)
            elapsed_secs = int(elapsed_seconds % 60)
            elapsed_time = f"{elapsed_minutes}m {elapsed_secs}s"
            
            # Ước tính thời gian còn lại (giả sử mỗi stage ~5 phút)
            # Stage format: "Stage X/6: ..."
            try:
                stage_text = current_job['stage']
                if 'Stage' in stage_text:
                    current_stage_num = int(stage_text.split('/')[0].split()[-1])
                    remaining_stages = 6 - current_stage_num
                    estimated_mins = remaining_stages * 5
                    estimated_remaining = f"~{estimated_mins} phút"
            except:
                estimated_remaining = "Đang tính toán..."
        
        status_info = {
            'server_status': 'Đang xử lý',
            'job_id': current_job['job_id'],
            'status': current_job['status'],
            'current_stage': current_job['stage'],
            'started_at': current_job['started_at'],
            'elapsed_time': elapsed_time,
            'estimated_remaining': estimated_remaining
        }
        
        if current_job['status'] == 'completed':
            status_info['server_status'] = 'Hoàn thành'
            status_info['completed_at'] = current_job['completed_at']
            status_info['message'] = f'Job {current_job["job_id"]} đã hoàn thành. Sử dụng GET /api/result/{current_job["job_id"]} để tải kết quả.'
            status_info.pop('estimated_remaining', None)
        
        elif current_job['status'] == 'failed':
            status_info['server_status'] = 'Lỗi'
            status_info['error'] = current_job['error']
            status_info['completed_at'] = current_job['completed_at']
            status_info.pop('estimated_remaining', None)
        
        return jsonify(status_info)

if __name__ == '__main__':
    # Cleanup at startup
    logger.info("🧹 Cleaning up old jobs at startup...")
    cleanup_old_jobs()
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Server starting on port {port}")
    logger.info(f"📍 Single worker mode - processing one job at a time")
    
    # Chạy với threaded=True để xử lý request status trong khi processing
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
