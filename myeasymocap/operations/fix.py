'''
import os
import json
import numpy as np
from os.path import join


class FixParams:
    """
    Module để fix params bằng cách thay thế poses từ kết quả HybrIK
    """
    def __init__(self, hybrik_res_dir='HybrIK/res', output_dir='fix', **kwargs):
        """
        Args:
            hybrik_res_dir: đường dẫn đến thư mục chứa kết quả HybrIK
            output_dir: đường dẫn để lưu params đã fix  
        """
        self.hybrik_res_dir = hybrik_res_dir
        self.output_dir = output_dir
        
        # Tạo thư mục output nếu chưa có
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fix_single_param(self, param, imgname):
        """
        Fix params cho một ảnh
        
        Args:
            param: dict chứa params từ infer module
            imgname: tên file ảnh
            
        Returns:
            dict chứa params đã được fix
        """
        # Bước 1: Copy params hiện tại
        current_params = param.copy()
        
        # Bước 2: Lấy tên file không có extension
        if isinstance(imgname, str):
            base_name = os.path.splitext(os.path.basename(imgname))[0]
        else:
            # Nếu imgname là list, lấy phần tử đầu tiên
            base_name = os.path.splitext(os.path.basename(imgname[0]))[0]
        
        # Đường dẫn đến file HybrIK result
        hybrik_json_path = join(self.hybrik_res_dir, f'{base_name}.json')
        
        if os.path.exists(hybrik_json_path):
            # Đọc poses từ HybrIK result
            with open(hybrik_json_path, 'r') as f:
                hybrik_data = json.load(f)
            
            # Thay thế poses
            if 'poses' in hybrik_data:
                # Convert list sang numpy array nếu cần
                poses = hybrik_data['poses']
                if isinstance(poses, list):
                    poses = np.array(poses)
                current_params['poses'] = poses
            else:
                print(f"⚠ Warning: 'poses' not found in {hybrik_json_path}")
        else:
            print(f"⚠ Warning: HybrIK result not found at {hybrik_json_path}")
            print(f"  Using original poses from infer module")
        
        # Bước 3: Lưu params đã fix vào file
        output_path = join(self.output_dir, f'{base_name}.json')
        
        # Convert numpy arrays sang list để có thể serialize JSON
        params_to_save = {}
        for key, value in current_params.items():
            if isinstance(value, np.ndarray):
                params_to_save[key] = value.tolist()
            elif isinstance(value, list):
                # Nếu là list, check từng phần tử
                params_to_save[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v 
                    for v in value
                ]
            else:
                params_to_save[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        
        return current_params
    
    def __call__(self, params, imgnames, **kwargs):
        """
        Method được gọi khi sử dụng đối tượng như một hàm
        
        Args:
            params: dict hoặc list dict chứa params từ infer module
                    format: {'Rh': [...], 'Th': [...], 'poses': [...], 'shapes': [...]}
            imgnames: str hoặc list str - tên file ảnh tương ứng với mỗi params
            **kwargs: các tham số khác (sẽ được giữ lại và trả về)
            
        Returns:
            dict chứa tất cả kwargs và 'params' đã được fix
        """
        # Xử lý params - có thể là list (nhiều người) hoặc dict (1 người)
        if isinstance(params, dict):
            params_list = [params]
            imgnames_list = [imgnames] if isinstance(imgnames, str) else imgnames
        else:
            params_list = params
            imgnames_list = imgnames if isinstance(imgnames, list) else [imgnames]
            
        fixed_params_list = []
        
        # Fix từng param
        for param, imgname in zip(params_list, imgnames_list):
            fixed_param = self.fix_single_param(param, imgname)
            fixed_params_list.append(fixed_param)
                
        # Trả về cùng format như input, cùng với tất cả kwargs
        result = kwargs.copy()
        if isinstance(params, dict):
            result['params'] = fixed_params_list[0]
        else:
            result['params'] = fixed_params_list
        
        return result
'''
import os
import json
import numpy as np
from os.path import join


class FixParams:
    """
    Module để fix params bằng cách thay thế poses từ kết quả HybrIK
    """
    def __init__(self, hybrik_res_dir='HybrIK/res', output_dir='fix', 
                 selective_joints=[16, 17, 18, 19], **kwargs):
        """
        Args:
            hybrik_res_dir: đường dẫn đến thư mục chứa kết quả HybrIK
            output_dir: đường dẫn để lưu params đã fix
            selective_joints: list các joint IDs cần thay thế (e.g., [16, 17, 18, 19])
                            Nếu None, thay thế toàn bộ poses
        """
        self.hybrik_res_dir = hybrik_res_dir
        self.output_dir = output_dir
        self.selective_joints = selective_joints
        
        # Tạo thư mục output nếu chưa có
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_joint_indices(self, joint_ids):
        """
        Chuyển đổi joint IDs sang indices trong poses array
        
        Args:
            joint_ids: list các joint IDs (1-based, e.g., [16, 17, 18, 19])
            
        Returns:
            list các indices trong poses array (0-based)
        """
        indices = []
        for joint_id in joint_ids:
            # Mỗi joint có 3 giá trị (rotation), joint_id bắt đầu từ 1
            start_idx = (joint_id - 1) * 3
            indices.extend([start_idx, start_idx + 1, start_idx + 2])
        return indices
        
    def fix_single_param(self, param, imgname):
        """
        Fix params cho một ảnh
        
        Args:
            param: dict chứa params từ infer module
            imgname: tên file ảnh
            
        Returns:
            dict chứa params đã được fix
        """
        # Bước 1: Copy params hiện tại
        current_params = param.copy()
        
        # Bước 2: Lấy tên file không có extension
        if isinstance(imgname, str):
            base_name = os.path.splitext(os.path.basename(imgname))[0]
        else:
            # Nếu imgname là list, lấy phần tử đầu tiên
            base_name = os.path.splitext(os.path.basename(imgname[0]))[0]
        
        # Đường dẫn đến file HybrIK result
        hybrik_json_path = join(self.hybrik_res_dir, f'{base_name}.json')
        
        if os.path.exists(hybrik_json_path):
            # Đọc poses từ HybrIK result
            with open(hybrik_json_path, 'r') as f:
                hybrik_data = json.load(f)
            
            # Thay thế poses
            if 'poses' in hybrik_data:
                # Convert list sang numpy array nếu cần
                hybrik_poses = hybrik_data['poses']
                if isinstance(hybrik_poses, list):
                    hybrik_poses = np.array(hybrik_poses)
                
                # Kiểm tra xem có thay thế selective hay toàn bộ
                if self.selective_joints is not None:
                    # Thay thế chỉ các joints được chỉ định
                    current_poses = current_params['poses'].copy()
                    if isinstance(current_poses, list):
                        current_poses = np.array(current_poses)
                    
                    # Lấy indices cần thay thế
                    indices = self._get_joint_indices(self.selective_joints)
                    
                    # Thay thế từng index
                    for idx in indices:
                        if idx < len(hybrik_poses) and idx < len(current_poses):
                            current_poses[idx] = hybrik_poses[idx]
                    
                    current_params['poses'] = current_poses
                    print(f"✓ Replaced joints {self.selective_joints} for {base_name}")
                else:
                    # Thay thế toàn bộ poses
                    current_params['poses'] = hybrik_poses
                    print(f"✓ Replaced all poses for {base_name}")
            else:
                print(f"⚠ Warning: 'poses' not found in {hybrik_json_path}")
        else:
            print(f"⚠ Warning: HybrIK result not found at {hybrik_json_path}")
            print(f"  Using original poses from infer module")
        
        # Bước 3: Lưu params đã fix vào file
        output_path = join(self.output_dir, f'{base_name}.json')
        
        # Convert numpy arrays sang list để có thể serialize JSON
        params_to_save = {}
        for key, value in current_params.items():
            if isinstance(value, np.ndarray):
                params_to_save[key] = value.tolist()
            elif isinstance(value, list):
                # Nếu là list, check từng phần tử
                params_to_save[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v 
                    for v in value
                ]
            else:
                params_to_save[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        
        return current_params
    
    def __call__(self, params, imgnames, **kwargs):
        """
        Method được gọi khi sử dụng đối tượng như một hàm
        
        Args:
            params: dict hoặc list dict chứa params từ infer module
                    format: {'Rh': [...], 'Th': [...], 'poses': [...], 'shapes': [...]}
            imgnames: str hoặc list str - tên file ảnh tương ứng với mỗi params
            **kwargs: các tham số khác (sẽ được giữ lại và trả về)
            
        Returns:
            dict chứa tất cả kwargs và 'params' đã được fix
        """
        # Xử lý params - có thể là list (nhiều người) hoặc dict (1 người)
        if isinstance(params, dict):
            params_list = [params]
            imgnames_list = [imgnames] if isinstance(imgnames, str) else imgnames
        else:
            params_list = params
            imgnames_list = imgnames if isinstance(imgnames, list) else [imgnames]
            
        fixed_params_list = []
        
        # Fix từng param
        for param, imgname in zip(params_list, imgnames_list):
            fixed_param = self.fix_single_param(param, imgname)
            fixed_params_list.append(fixed_param)
                
        # Trả về cùng format như input, cùng với tất cả kwargs
        result = kwargs.copy()
        if isinstance(params, dict):
            result['params'] = fixed_params_list[0]
        else:
            result['params'] = fixed_params_list
        
        return result