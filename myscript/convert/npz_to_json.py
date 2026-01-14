import numpy as np
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Convert .npz file to .json")
    parser.add_argument('-i', '--input', required=True, help="Input .npz file path")
    args = parser.parse_args()

    input_path = args.input

    if not os.path.isfile(input_path):
        print(f"File '{input_path}' không tồn tại.")
        return

    # Load file npz
    data = np.load(input_path)

    # data là dạng dict-like chứa các mảng
    # Chuyển tất cả sang list để json dump được
    data_dict = {key: data[key].tolist() for key in data.files}

    base_name = os.path.splitext(input_path)[0]
    output_path = base_name + ".json"

    with open(output_path, "w") as f:
        json.dump(data_dict, f, indent=4)

    # In thông tin các key và shape tương ứng
    print(f"Đã lưu dữ liệu vào {output_path}")
    for key in data.files:
        print(f"{key}: shape = {data[key].shape}")

if __name__ == "__main__":
    main()
