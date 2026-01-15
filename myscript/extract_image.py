# extract image from videos
import os
from os.path import join
from glob import glob
import subprocess

extensions = ['.mp4', '.webm', '.flv', '.MP4', '.MOV', '.mov', '.avi']

def run(cmd):
    os.system(cmd)

def get_video_frame_count(video_path, ffmpeg='ffmpeg'):
    """Lấy tổng số frame của video"""
    try:
        cmd = f'{ffmpeg} -i "{video_path}" -map 0:v:0 -c copy -f null - 2>&1'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stderr
        
        for line in output.split('\n'):
            if 'frame=' in line:
                parts = line.split('frame=')
                if len(parts) > 1:
                    frame_str = parts[1].split()[0].strip()
                    try:
                        return int(frame_str)
                    except:
                        pass
        return None
    except:
        return None

def extract_images(input_folder, output_folder, ffmpeg='ffmpeg', num=-1, scale=1, transpose=-1, remove=0, restart=False, debug=False):
    """
    Trích xuất ảnh từ các video trong input_folder
    Mỗi video sẽ tạo thư mục riêng trong output_folder
    """
    # Tìm tất cả video trong input_folder
    videos = sorted(sum([
        glob(join(input_folder, '*'+ext)) for ext in extensions
        ], [])
    )
    
    if not videos:
        print(f"❌ Không tìm thấy video nào trong folder: {input_folder}")
        return
    
    print("\n" + "="*70)
    print("📂 INPUT")
    print("="*70)
    print(f"   Folder: {os.path.abspath(input_folder)}")
    print(f"   Số video: {len(videos)}")
    for i, v in enumerate(videos, 1):
        print(f"   [{i}] {os.path.basename(v)}")
    
    print("\n" + "="*70)
    print("📁 OUTPUT")
    print("="*70)
    print(f"   Folder: {os.path.abspath(output_folder)}")
    print(f"   Cấu trúc: output/<tên_video>/images/")
    
    print("\n" + "="*70)
    print("⚙️  PROCESS")
    print("="*70)
    
    for idx, videoname in enumerate(videos, 1):
        # Lấy tên file không có phần mở rộng
        video_basename = '.'.join(os.path.basename(videoname).split('.')[:-1])
        
        # Tạo thư mục output cho video này: output/tên_video/images/
        outpath = join(output_folder, video_basename, 'images')
        
        # Kiểm tra nếu đã tồn tại và có đủ ảnh
        if os.path.exists(outpath) and (len(os.listdir(outpath)) > 10 or (num != -1 and len(os.listdir(outpath)) == num)) and not restart:
            print(f"\n[{idx}/{len(videos)}] ⏭️  {video_basename}")
            print(f"        Bỏ qua - đã tồn tại {len(os.listdir(outpath))} ảnh")
            continue
        
        os.makedirs(outpath, exist_ok=True)
        
        # Xây dựng câu lệnh ffmpeg
        other_cmd = ''
        if num != -1:
            other_cmd += '-vframes {}'.format(num)
        
        if scale != 1 and transpose != -1:
            other_cmd += ' -vf "transpose={transpose},scale=iw/{scale}:ih/{scale}"'.format(scale=scale, transpose=transpose)
        elif scale != 1:
            other_cmd += ' -vf "scale=iw/{scale}:ih/{scale}"'.format(scale=scale)
        elif transpose != -1:
            other_cmd += ' -vf transpose={}'.format(transpose)
        
        # Thêm progress vào cmd
        cmd = '{} -i "{}" {} -q:v 1 -start_number 0 -progress pipe:1 "{}/%06d.jpg"'.format(
            ffmpeg, videoname, other_cmd, outpath)
        
        if not debug:
            cmd += ' -loglevel error'
        
        print(f"\n[{idx}/{len(videos)}] 🎬 {video_basename}")
        print(f"        Output: {outpath}")
        
        # Chạy ffmpeg và hiển thị progress
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        total_frames = get_video_frame_count(videoname, ffmpeg) if num == -1 else num
        current_frame = 0
        
        for line in process.stdout:
            if 'frame=' in line:
                try:
                    frame_num = int(line.split('=')[1].strip())
                    current_frame = frame_num
                    if total_frames:
                        percent = min(100, (current_frame / total_frames) * 100)
                        bar_length = 30
                        filled = int(bar_length * percent / 100)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f"\r        [{bar}] {percent:.1f}% ({current_frame}/{total_frames} frames)", end='', flush=True)
                    else:
                        print(f"\r        Processing... {current_frame} frames", end='', flush=True)
                except:
                    pass
        
        process.wait()
        print()  # Xuống dòng sau khi hoàn thành
        
        # Xóa frame nếu cần
        if remove != 0:
            frames = sorted(glob(join(outpath, '*.jpg')))
            if remove > 0:
                # Xóa frame đầu
                print(f"        🗑️  Xóa {remove} frame đầu...")
                for i in range(min(remove, len(frames))):
                    os.remove(frames[i])
                # Đổi tên lại các frame còn lại
                remaining_frames = frames[remove:]
                for idx_frame, frame in enumerate(remaining_frames):
                    new_name_file = join(outpath, f"{idx_frame:06d}.jpg")
                    os.rename(frame, new_name_file)
            elif remove < 0:
                # Xóa frame cuối
                print(f"        🗑️  Xóa {abs(remove)} frame cuối...")
                frames_to_remove = frames[remove:] 
                for frame in frames_to_remove:
                    os.remove(frame)
        
        final_count = len(glob(join(outpath, '*.jpg')))
        print(f"        ✅ Hoàn thành: {final_count} ảnh")

if __name__ == "__main__":
    # Cấu hình cố định
    INPUT_FOLDER = 'input'      # Folder chứa các video
    OUTPUT_FOLDER = 'output'    # Folder chứa các thư mục ảnh đã tách
    
    # Các tham số mặc định
    FFMPEG = 'ffmpeg'
    NUM_FRAMES = -1             # -1 = lấy tất cả frames
    SCALE = 1                   # 1 = không thu nhỏ
    TRANSPOSE = -1              # -1 = không xoay
    REMOVE = 0                  # 0 = không xóa frame nào
    RESTART = False             # False = bỏ qua video đã xử lý
    DEBUG = False               # False = không hiển thị log ffmpeg
    
    # Tạo folder input nếu chưa có
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("\n" + "="*70)
    print("🎥 TRÍCH XUẤT ẢNH TỪ VIDEO")
    print("="*70)
    
    # Chạy trích xuất
    extract_images(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        ffmpeg=FFMPEG,
        num=NUM_FRAMES,
        scale=SCALE,
        transpose=TRANSPOSE,
        remove=REMOVE,
        restart=RESTART,
        debug=DEBUG
    )
    print("HOÀN THÀNH!")
