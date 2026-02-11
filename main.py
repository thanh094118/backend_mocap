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

# --- SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Global state
current_job = {
    'job_id': None,
    'status': 'idle',
    'stage': '',
    'error': None,
    'result_path': None
}
job_lock = threading.Lock()

TEMP_DIR = Path('/app/temp')
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# --- HELPER FUNCTIONS ---

def cleanup_all_temp():
    """Xóa SẠCH thư mục temp để tránh tràn data"""
    try:
        # Xóa nội dung trong TEMP_DIR nhưng giữ lại folder gốc
        for item in TEMP_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        logger.info("♻️  Cleaned up workspace before new job")
    except Exception as e:
        logger.error(f"⚠️ Cleanup error: {e}")

def run_command(command, cwd, stage_name):
    """Hàm chạy lệnh chung để tối giản code"""
    try:
        logger.info(f"⏳ [{stage_name}] Processing...")
        # Capture output=False để không spam log, chỉ check returncode
        result = subprocess.run(
            command, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            timeout=3600
        )
        if result.returncode != 0:
            # Chỉ lấy dòng lỗi cuối cùng hoặc tin nhắn lỗi ngắn gọn
            error_msg = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown error"
            raise Exception(f"Exit code {result.returncode}: {error_msg}")
            
        logger.info(f"✅ [{stage_name}] Completed")
        return True
    except subprocess.TimeoutExpired:
        raise Exception("Timeout")
    except Exception as e:
        raise e

def run_pipeline(job_id, video1_path, video2_path, work_dir):
    global current_job
    work_dir = Path(work_dir)
    
    try:
        # --- Stage 1: Extract ---
        stage = 'Stage 1/6: Extract images'
        with job_lock: current_job['stage'] = stage
        
        run_command(
            ['python', '/app/myscript/extract_image.py', video1_path, video2_path],
            cwd=work_dir, stage_name='Extract Images'
        )

        # --- Stage 2: Camera 1 ---
        stage = 'Stage 2/6: Processing Camera 1'
        with job_lock:
            current_job['stage'] = stage
        
        camera1_out = work_dir / 'output' / 'camera1'
        
        try:
            print("\n" + "="*80)
            print(f"[START] {stage}")
            print("Working directory:", work_dir)
            print("Output path:", camera1_out)
            print("="*80)
        
            run_command(
                [
                    'emc',
                    '--data', '/app/config/datasets/svimage.yml',
                    '--exp', '/app/config/1v1p/hrnet_pare_finetune.yml',
                    '--root', str(camera1_out),
                    '--subs', 'images'
                ],
                cwd=work_dir,
                stage_name='Camera 1'
            )
        
            print(f"[SUCCESS] {stage} finished")
        
        except Exception as e:
            import traceback
            print("\n" + "="*80)
            print(f"[ERROR] {stage} crashed")
            print("Error type:", type(e))
            print("Error message:", str(e))
            print("\nFull traceback:")
            traceback.print_exc()
            print("="*80)
        
            raise  # giữ nguyên behavior để job fail

        # --- Stage 3: Camera 2 ---
        stage = 'Stage 3/6: Processing Camera 2'
        with job_lock: current_job['stage'] = stage
        
        camera2_out = work_dir / 'output' / 'camera2'
        run_command(
            ['emc', '--data', '/app/config/datasets/svimage.yml', 
             '--exp', '/app/config/1v1p/hrnet_pare_finetune.yml', 
             '--root', str(camera2_out), '--subs', 'images'],
            cwd=work_dir, stage_name='Camera 2'
        )

        # --- Stage 4: Merge Poses ---
        stage = 'Stage 4/6: Merging Poses'
        with job_lock: current_job['stage'] = stage
        
        run_command(
            ['python', '/app/myscript/merged_poses.py'],
            cwd=work_dir, stage_name='Merge Poses'
        )

        # --- Stage 5: Pixel Processing ---
        stage = 'Stage 5/6: Pixel Processing'
        with job_lock: current_job['stage'] = stage
        
        run_command(
            ['python', '/app/myscript/pixel.py'],
            cwd=work_dir, stage_name='Pixel Process'
        )

        # --- Stage 6: Final & Align ---
        stage = 'Stage 6/6: Final Processing'
        with job_lock: current_job['stage'] = stage
        
        run_command(
            ['python', '/app/myscript/merged_total.py'],
            cwd=work_dir, stage_name='Merge Total'
        )
        run_command(
            ['python', '/app/myscript/aligned.py'],
            cwd=work_dir, stage_name='Align'
        )

        # --- Packaging - Nén cả 2 folder ---
        logger.info("📦 Zipping results...")
        output_dir = work_dir / 'output'
        output_final_folder = output_dir / 'output_final'
        merged_total_folder = output_dir / 'merged_total'
        zip_path = work_dir / 'results.zip'
        
        # Kiểm tra cả 2 folder có tồn tại
        if not output_final_folder.exists():
            raise Exception("output_final folder missing")
        if not merged_total_folder.exists():
            raise Exception("merged_total folder missing")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Nén folder output_final
            for file_path in output_final_folder.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, Path('output_final') / file_path.relative_to(output_final_folder))
            
            # Nén folder merged_total
            for file_path in merged_total_folder.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, Path('merged_total') / file_path.relative_to(merged_total_folder))

        with job_lock:
            current_job['status'] = 'completed'
            current_job['stage'] = 'Finished'
            current_job['result_path'] = str(zip_path)
        
        logger.info(f"🎉 Job {job_id} Finished Successfully")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Job Failed at {current_job['stage']}: {error_msg}")
        with job_lock:
            current_job['status'] = 'failed'
            current_job['error'] = error_msg

# --- API ENDPOINTS ---

@app.route('/api/upload', methods=['POST'])
def process_videos():
    global current_job
    
    try:
        # 1. CLEANUP FIRST: Xóa mọi thứ cũ trước khi làm gì hết
        cleanup_all_temp()

        # 2. Validate inputs
        if 'video1' not in request.files or 'video2' not in request.files:
            return jsonify({'error': 'Missing video files'}), 400
        
        video1 = request.files['video1']
        video2 = request.files['video2']
        
        # 3. Setup New Job
        job_id = str(uuid.uuid4())
        work_dir = TEMP_DIR / job_id
        input_dir = work_dir / 'input'
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. Save Files
        v1_path = input_dir / 'camera1.mp4'
        v2_path = input_dir / 'camera2.mp4'
        video1.save(str(v1_path))
        video2.save(str(v2_path))
        
        # 5. Update State & Start Thread
        with job_lock:
            current_job = {
                'job_id': job_id,
                'status': 'processing',
                'stage': 'Initializing',
                'error': None,
                'result_path': None
            }
        
        logger.info(f"🚀 New Job Started: {job_id}")
        threading.Thread(
            target=run_pipeline,
            args=(job_id, str(v1_path), str(v2_path), str(work_dir)),
            daemon=True
        ).start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'message': 'Job started'
        }), 202

    except Exception as e:
        logger.error(f"❌ API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Tối giản thông tin status trả về"""
    with job_lock:
        response = {
            'job_id': current_job['job_id'],
            'status': current_job['status'],
            'stage': current_job['stage']
        }
        if current_job['status'] == 'failed':
            response['error'] = current_job['error']
            
        return jsonify(response)

@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    with job_lock:
        if current_job['job_id'] != job_id:
            return jsonify({'error': 'Job ID mismatch or expired'}), 404
        
        if current_job['status'] == 'processing':
            return jsonify({'status': 'processing'}), 202
            
        if current_job['status'] == 'failed':
            return jsonify({'status': 'failed', 'error': current_job['error']}), 500
            
        if current_job['status'] == 'completed' and current_job['result_path']:
            return send_file(
                current_job['result_path'],
                as_attachment=True,
                download_name=f'result_{job_id}.zip'
            )
            
    return jsonify({'error': 'Unknown state'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Chạy cleanup lần đầu khi khởi động server
    cleanup_all_temp()
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
