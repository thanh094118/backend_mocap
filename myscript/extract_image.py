# extract image from videos
import os
from os.path import join
from glob import glob

extensions = ['.mp4', '.webm', '.flv', '.MP4', '.MOV', '.mov', '.avi']

def run(cmd):
    print(cmd)
    os.system(cmd)

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
        print(f"Không tìm thấy video nào trong folder: {input_folder}")
        return
    
    print(f"Tìm thấy {len(videos)} video(s)")
    
    for videoname in videos:
        # Lấy tên file không có phần mở rộng
        video_basename = '.'.join(os.path.basename(videoname).split('.')[:-1])
        
        # Tạo thư mục output cho video này: output/tên_video/images/
        outpath = join(output_folder, video_basename, 'images')
        
        # Kiểm tra nếu đã tồn tại và có đủ ảnh
        if os.path.exists(outpath) and (len(os.listdir(outpath)) > 10 or (num != -1 and len(os.listdir(outpath)) == num)) and not restart:
            print(f"Bỏ qua {video_basename} - đã tồn tại")
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
        
        cmd = '{} -i "{}" {} -q:v 1 -start_number 0 "{}/%06d.jpg"'.format(
            ffmpeg, videoname, other_cmd, outpath)
        
        if not debug:
            cmd += ' -loglevel quiet'
        
        print(f"\nĐang xử lý: {video_basename}")
        run(cmd)
        
        # Xóa frame nếu cần
        if remove != 0:
            frames = sorted(glob(join(outpath, '*.jpg')))
            if remove > 0:
                # Xóa frame đầu
                for i in range(min(remove, len(frames))):
                    os.remove(frames[i])
                    print(f"Đã xóa: {frames[i]}")
                # Đổi tên lại các frame còn lại
                remaining_frames = frames[remove:]
                for idx, frame in enumerate(remaining_frames):
                    new_name_file = join(outpath, f"{idx:06d}.jpg")
                    os.rename(frame, new_name_file)
            elif remove < 0:
                # Xóa frame cuối
                frames_to_remove = frames[remove:] 
                for frame in frames_to_remove:
                    os.remove(frame)
                    print(f"Đã xóa: {frame}")
        
        print(f"Hoàn thành: {video_basename}")

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
    
    print("=" * 50)
    print("TRÍCH XUẤT ẢNH TỪ VIDEO")
    print("=" * 50)
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("=" * 50)
    
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
    
    print("\n" + "=" * 50)
    print("HOÀN THÀNH!")
    print("=" * 50)