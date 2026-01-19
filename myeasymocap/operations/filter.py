"""
Module lọc keypoints 2D có độ tin cậy thấp
Path: myeasymocap/operations/filter.py
"""
import numpy as np
import os
import json


class FilterKeypoints:
    """
    Lọc các keypoints 2D có độ tin cậy thấp để giảm nhiễu trong quá trình tối ưu
    
    Args:
        confidence_threshold (float): Ngưỡng confidence. Keypoints có conf < threshold sẽ bị set confidence = 0
        low_confidence_display_threshold (float): Ngưỡng để hiển thị các keypoints có confidence thấp (mặc định: 0.5)
        save_low_confidence (bool): Có lưu thông tin low confidence keypoints vào file JSON hay không (mặc định: True)
        name (str): Tên thư mục con để lưu file JSON (mặc định: 'confident')
    """
    
    def __init__(self, confidence_threshold=0.3, low_confidence_display_threshold=0.5, 
                 save_low_confidence=True, name='confident'):
        self.confidence_threshold = confidence_threshold
        self.low_confidence_display_threshold = low_confidence_display_threshold
        self.save_low_confidence = save_low_confidence
        self.output = '/tmp'  # Sẽ được set từ pipeline
        self.name = name
        self.skip_joint_ids = {19, 20, 21, 22, 23, 24}  # Bỏ qua các joint này
        
        print(f"[FilterKeypoints] Initialized with confidence_threshold: {self.confidence_threshold}")
        print(f"[FilterKeypoints] Low confidence display threshold: {self.low_confidence_display_threshold}")
        if self.save_low_confidence:
            print(f"[FilterKeypoints] Will save low confidence data to: {{output}}/{self.name}")
            print(f"[FilterKeypoints] Skipping joint IDs: {sorted(self.skip_joint_ids)}")
    
    def __call__(self, keypoints, images=None, imgnames=None, **kwargs):
        """
        Args:
            keypoints: np.array có thể có nhiều shape khác nhau:
                - (J, 3): single frame, single person
                - (nViews, J, 3): multi-view, single person
                - List[np.array(nPersons, J, 3)]: multi-view, multi-person
            images: Danh sách ảnh (không bắt buộc)
            imgnames: Danh sách tên file ảnh để đặt tên cho JSON output
        
        Returns:
            dict: {'keypoints': filtered_keypoints} với cùng cấu trúc
        """
        # Xử lý dựa trên cấu trúc dữ liệu
        if isinstance(keypoints, np.ndarray):
            if keypoints.ndim == 2:
                # Shape (J, 3) - single frame
                filtered_keypoints = self._filter_array_2d(keypoints, imgnames)
            elif keypoints.ndim == 3:
                # Shape (nViews, J, 3) - multi-view
                filtered_keypoints = self._filter_array_3d(keypoints, imgnames)
            else:
                raise ValueError(f"Unexpected keypoints shape: {keypoints.shape}")
        elif isinstance(keypoints, list):
            # List[np.array] - multiple persons
            filtered_keypoints = self._filter_list(keypoints, imgnames)
        else:
            raise TypeError(f"Unsupported keypoints type: {type(keypoints)}")
        
        return {'keypoints': filtered_keypoints}
    
    def _get_output_filename(self, imgnames, view_idx=None, person_idx=None):
        """
        Tạo tên file output dựa trên imgnames
        """
        if imgnames is None:
            return None
        
        # Lấy tên file từ imgnames
        if isinstance(imgnames, list):
            if view_idx is not None and view_idx < len(imgnames):
                imgname = imgnames[view_idx]
            else:
                imgname = imgnames[0] if len(imgnames) > 0 else None
        else:
            imgname = imgnames
        
        if imgname is None:
            return None
        
        # Lấy basename và bỏ extension
        basename = os.path.splitext(os.path.basename(imgname))[0]
        
        # Thêm suffix nếu có person_idx
        if person_idx is not None:
            basename = f"{basename}_person{person_idx}"
        
        # Tạo đường dẫn đầy đủ giống như Vis2D
        outname = os.path.join(self.output, self.name, f"{basename}.json")
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        
        return outname
    
    def _save_low_confidence_data(self, low_conf_data, output_file):
        """
        Lưu thông tin low confidence keypoints vào file JSON
        """
        if output_file is None or not self.save_low_confidence:
            return
        
        try:
            with open(output_file, 'w') as f:
                json.dump(low_conf_data, f, indent=2)
            print(f"[FilterKeypoints] Saved low confidence data to: {output_file}")
        except Exception as e:
            print(f"[FilterKeypoints] Error saving to {output_file}: {e}")
    
    def _filter_array_2d(self, keypoints, imgnames=None):
        """
        Lọc keypoints shape (J, 3)
        """
        nJoints = keypoints.shape[0]
        filtered_kpts = keypoints.copy()
        confidences = filtered_kpts[:, 2]
        
        # Thu thập keypoints có confidence < low_confidence_display_threshold
        low_display_mask = confidences < self.low_confidence_display_threshold
        low_conf_data = []
        
        if np.any(low_display_mask):
            low_conf_indices = np.where(low_display_mask)[0]
            print(f"[FilterKeypoints] Keypoints với confidence < {self.low_confidence_display_threshold}:")
            for idx in low_conf_indices:
                # Bỏ qua các joint IDs không cần thiết
                if idx in self.skip_joint_ids:
                    continue
                    
                kpt_info = {
                    'joint_id': int(idx),
                    'confidence': float(confidences[idx])
                }
                low_conf_data.append(kpt_info)
                print(f"  - Joint {idx}: conf={confidences[idx]:.3f}, pos=({keypoints[idx, 0]:.2f}, {keypoints[idx, 1]:.2f})")
        
        # Lưu vào file JSON
        if self.save_low_confidence and len(low_conf_data) > 0:
            output_file = self._get_output_filename(imgnames)
            if output_file:
                self._save_low_confidence_data(low_conf_data, output_file)
        
        # Lọc: set confidence = 0 cho keypoints có conf < threshold
        low_conf_mask = confidences < self.confidence_threshold
        filtered_kpts[low_conf_mask, 2] = 0.0
        
        n_filtered = np.sum(low_conf_mask)
        n_valid = nJoints - n_filtered
        avg_conf = confidences[confidences > 0].mean() if np.any(confidences > 0) else 0.0
                
        return filtered_kpts
    
    def _filter_array_3d(self, keypoints, imgnames=None):
        """
        Lọc keypoints shape (nViews, J, 3)
        """
        nViews, nJoints, _ = keypoints.shape
        filtered_kpts = keypoints.copy()
        
        total_filtered = 0
        for view_idx in range(nViews):
            confidences = filtered_kpts[view_idx, :, 2]
            
            # Thu thập keypoints có confidence < low_confidence_display_threshold
            low_display_mask = confidences < self.low_confidence_display_threshold
            low_conf_data = []
            
            if np.any(low_display_mask):
                low_conf_indices = np.where(low_display_mask)[0]
                print(f"[FilterKeypoints] View {view_idx} - Keypoints với confidence < {self.low_confidence_display_threshold}:")
                for idx in low_conf_indices:
                    # Bỏ qua các joint IDs không cần thiết
                    if idx in self.skip_joint_ids:
                        continue
                        
                    kpt_info = {
                        'joint_id': int(idx),
                        'confidence': float(confidences[idx])
                    }
                    low_conf_data.append(kpt_info)
                    print(f"  - Joint {idx}: conf={confidences[idx]:.3f}, pos=({keypoints[view_idx, idx, 0]:.2f}, {keypoints[view_idx, idx, 1]:.2f})")
            
            # Lưu vào file JSON
            if self.save_low_confidence and len(low_conf_data) > 0:
                output_file = self._get_output_filename(imgnames, view_idx=view_idx)
                if output_file:
                    self._save_low_confidence_data(low_conf_data, output_file)
            
            # Lọc: set confidence = 0 cho keypoints có conf < threshold
            low_conf_mask = confidences < self.confidence_threshold
            filtered_kpts[view_idx, low_conf_mask, 2] = 0.0
            
            total_filtered += np.sum(low_conf_mask)
        
        avg_conf = filtered_kpts[:, :, 2][filtered_kpts[:, :, 2] > 0].mean() if np.any(filtered_kpts[:, :, 2] > 0) else 0.0
        print(f"[FilterKeypoints] {nViews} views: filtered {total_filtered}/{nViews*nJoints} keypoints (avg conf: {avg_conf:.3f})")
        
        return filtered_kpts
    
    def _filter_list(self, keypoints_list, imgnames=None):
        """
        Lọc keypoints dạng List[np.array(nPersons, J, 3)]
        """
        filtered_list = []
        nViews = len(keypoints_list)
        
        total_filtered = 0
        total_keypoints = 0
        
        for view_idx, view_kpts in enumerate(keypoints_list):
            if view_kpts.shape[0] == 0:
                # Không có người trong view này
                filtered_list.append(view_kpts.copy())
                continue
            
            nPersons, nJoints, _ = view_kpts.shape
            filtered_view = view_kpts.copy()
            
            for person_idx in range(nPersons):
                confidences = filtered_view[person_idx, :, 2]
                
                # Thu thập keypoints có confidence < low_confidence_display_threshold
                low_display_mask = confidences < self.low_confidence_display_threshold
                low_conf_data = []
                
                if np.any(low_display_mask):
                    low_conf_indices = np.where(low_display_mask)[0]
                    print(f"[FilterKeypoints] View {view_idx}, Person {person_idx} - Keypoints với confidence < {self.low_confidence_display_threshold}:")
                    for idx in low_conf_indices:
                        # Bỏ qua các joint IDs không cần thiết
                        if idx in self.skip_joint_ids:
                            continue
                            
                        kpt_info = {
                            'joint_id': int(idx),
                            'confidence': float(confidences[idx])
                        }
                        low_conf_data.append(kpt_info)
                        print(f"  - Joint {idx}: conf={confidences[idx]:.3f}, pos=({view_kpts[person_idx, idx, 0]:.2f}, {view_kpts[person_idx, idx, 1]:.2f})")
                
                # Lưu vào file JSON
                if self.save_low_confidence and len(low_conf_data) > 0:
                    output_file = self._get_output_filename(imgnames, view_idx=view_idx, person_idx=person_idx)
                    if output_file:
                        self._save_low_confidence_data(low_conf_data, output_file)
                
                # Lọc: set confidence = 0 cho keypoints có conf < threshold
                low_conf_mask = confidences < self.confidence_threshold
                filtered_view[person_idx, low_conf_mask, 2] = 0.0
                
                total_filtered += np.sum(low_conf_mask)
                total_keypoints += nJoints
            
            filtered_list.append(filtered_view)
        
        print(f"[FilterKeypoints] {nViews} views: filtered {total_filtered}/{total_keypoints} keypoints")
        
        return filtered_list
