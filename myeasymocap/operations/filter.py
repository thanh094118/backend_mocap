"""
Module lọc keypoints 2D có độ tin cậy thấp
Path: myeasymocap/operations/filter.py
"""
import numpy as np


class FilterKeypoints:
    """
    Lọc các keypoints 2D có độ tin cậy thấp để giảm nhiễu trong quá trình tối ưu
    
    Args:
        confidence_threshold (float): Ngưỡng confidence. Keypoints có conf < threshold sẽ bị set confidence = 0
    """
    
    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        
        print(f"[FilterKeypoints] Initialized with confidence_threshold: {self.confidence_threshold}")
    
    def __call__(self, keypoints, **kwargs):
        """
        Args:
            keypoints: np.array có thể có nhiều shape khác nhau:
                - (J, 3): single frame, single person
                - (nViews, J, 3): multi-view, single person
                - List[np.array(nPersons, J, 3)]: multi-view, multi-person
        
        Returns:
            dict: {'keypoints': filtered_keypoints} với cùng cấu trúc
        """
        # Xử lý dựa trên cấu trúc dữ liệu
        if isinstance(keypoints, np.ndarray):
            if keypoints.ndim == 2:
                # Shape (J, 3) - single frame
                filtered_keypoints = self._filter_array_2d(keypoints)
            elif keypoints.ndim == 3:
                # Shape (nViews, J, 3) - multi-view
                filtered_keypoints = self._filter_array_3d(keypoints)
            else:
                raise ValueError(f"Unexpected keypoints shape: {keypoints.shape}")
        elif isinstance(keypoints, list):
            # List[np.array] - multiple persons
            filtered_keypoints = self._filter_list(keypoints)
        else:
            raise TypeError(f"Unsupported keypoints type: {type(keypoints)}")
        
        return {'keypoints': filtered_keypoints}
    
    def _filter_array_2d(self, keypoints):
        """
        Lọc keypoints shape (J, 3)
        """
        nJoints = keypoints.shape[0]
        filtered_kpts = keypoints.copy()
        confidences = filtered_kpts[:, 2]
        
        # Lọc: set confidence = 0 cho keypoints có conf < threshold
        low_conf_mask = confidences < self.confidence_threshold
        filtered_kpts[low_conf_mask, 2] = 0.0
        
        n_filtered = np.sum(low_conf_mask)
        n_valid = nJoints - n_filtered
        avg_conf = confidences[confidences > 0].mean() if np.any(confidences > 0) else 0.0
                
        return filtered_kpts
    
    def _filter_array_3d(self, keypoints):
        """
        Lọc keypoints shape (nViews, J, 3)
        """
        nViews, nJoints, _ = keypoints.shape
        filtered_kpts = keypoints.copy()
        
        total_filtered = 0
        for view_idx in range(nViews):
            confidences = filtered_kpts[view_idx, :, 2]
            
            # Lọc: set confidence = 0 cho keypoints có conf < threshold
            low_conf_mask = confidences < self.confidence_threshold
            filtered_kpts[view_idx, low_conf_mask, 2] = 0.0
            
            total_filtered += np.sum(low_conf_mask)
        
        avg_conf = filtered_kpts[:, :, 2][filtered_kpts[:, :, 2] > 0].mean() if np.any(filtered_kpts[:, :, 2] > 0) else 0.0
        print(f"[FilterKeypoints] {nViews} views: filtered {total_filtered}/{nViews*nJoints} keypoints (avg conf: {avg_conf:.3f})")
        
        return filtered_kpts
    
    def _filter_list(self, keypoints_list):
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
                
                # Lọc: set confidence = 0 cho keypoints có conf < threshold
                low_conf_mask = confidences < self.confidence_threshold
                filtered_view[person_idx, low_conf_mask, 2] = 0.0
                
                total_filtered += np.sum(low_conf_mask)
                total_keypoints += nJoints
            
            filtered_list.append(filtered_view)
        
        print(f"[FilterKeypoints] {nViews} views: filtered {total_filtered}/{total_keypoints} keypoints")
        
        return filtered_list