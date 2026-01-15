import numpy as np
import pickle
import torch

class RemoveHands:
    def __init__(self, partseg_path):
        # Load file phân vùng
        with open(partseg_path, 'rb') as f:
            mapping = pickle.load(f)
        
        self.part2num = mapping['part2num']
        self.smpl_index = mapping['smpl_index']
        
        # ID bàn tay
        self.left_hand_id = self.part2num.get('L_Hand', None)
        self.right_hand_id = self.part2num.get('R_Hand', None)
        if self.left_hand_id is None or self.right_hand_id is None:
            raise ValueError("Không tìm thấy ID bàn tay trong part2num")
        
        # ID cẳng tay
        self.left_forearm_id = self.part2num.get('L_ForeArm', None)
        self.right_forearm_id = self.part2num.get('R_ForeArm', None)
        if self.left_forearm_id is None or self.right_forearm_id is None:
            raise ValueError("Không tìm thấy ID cẳng tay trong part2num")
        
        # Danh sách index
        self.left_hand_vertices = np.where(self.smpl_index == self.left_hand_id)[0]
        self.right_hand_vertices = np.where(self.smpl_index == self.right_hand_id)[0]
        self.left_forearm_vertices = np.where(self.smpl_index == self.left_forearm_id)[0]
        self.right_forearm_vertices = np.where(self.smpl_index == self.right_forearm_id)[0]

    def __call__(self, **kwargs):
        """
        Xử lý kwargs với body_model và params
        """
        try:
            # Lấy body_model và params từ kwargs
            body_model = kwargs['body_model']
            params = kwargs['params']
            
            # Tính toán output từ body_model
            output = body_model(**params)
            
            # output là tensor với shape [batch_size, num_vertices, 3]
            vertices = output.clone()
            batch_size = vertices.shape[0]
            
            # Xử lý từng batch
            for b in range(batch_size):
                # Xử lý bàn tay trái
                if len(self.left_forearm_vertices) > 0 and len(self.left_hand_vertices) > 0:
                    left_forearm_coords = vertices[b, self.left_forearm_vertices, :]
                    if len(left_forearm_coords) > 0:
                        # Tìm điểm anchor (gần trục x=0 nhất)
                        left_anchor_idx = torch.argmin(torch.abs(left_forearm_coords[:, 0]))
                        left_anchor_pos = left_forearm_coords[left_anchor_idx:left_anchor_idx+1, :]
                        # Đặt tất cả vertices của bàn tay về vị trí anchor
                        vertices[b, self.left_hand_vertices, :] = left_anchor_pos
                
                # Xử lý bàn tay phải
                if len(self.right_forearm_vertices) > 0 and len(self.right_hand_vertices) > 0:
                    right_forearm_coords = vertices[b, self.right_forearm_vertices, :]
                    if len(right_forearm_coords) > 0:
                        # Tìm điểm anchor (gần trục x=0 nhất)
                        right_anchor_idx = torch.argmin(torch.abs(right_forearm_coords[:, 0]))
                        right_anchor_pos = right_forearm_coords[right_anchor_idx:right_anchor_idx+1, :]
                        # Đặt tất cả vertices của bàn tay về vị trí anchor
                        vertices[b, self.right_hand_vertices, :] = right_anchor_pos
            
            # Trả về dictionary với key 'vertices' để tương thích với hệ thống
            return {'vertices': vertices}
            
        except Exception as e:
            print(f"Error in RemoveHands: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: trả về output gốc dưới dạng dictionary
            if 'body_model' in kwargs and 'params' in kwargs:
                body_model = kwargs['body_model']
                params = kwargs['params']
                output = body_model(**params)
                return {'vertices': output}
            else:
                raise e


class RemoveHandsAdvanced:
    def __init__(self, partseg_path, shrink_factor=0.1):
        with open(partseg_path, 'rb') as f:
            mapping = pickle.load(f)
        
        self.part2num = mapping['part2num']
        self.smpl_index = mapping['smpl_index']
        self.shrink_factor = shrink_factor
        
        self.left_hand_id = self.part2num.get('L_Hand', None)
        self.right_hand_id = self.part2num.get('R_Hand', None)
        self.left_forearm_id = self.part2num.get('L_ForeArm', None)
        self.right_forearm_id = self.part2num.get('R_ForeArm', None)
        
        if None in [self.left_hand_id, self.right_hand_id, self.left_forearm_id, self.right_forearm_id]:
            raise ValueError("Không tìm thấy đầy đủ ID các bộ phận cần thiết")
        
        self.left_hand_vertices = np.where(self.smpl_index == self.left_hand_id)[0]
        self.right_hand_vertices = np.where(self.smpl_index == self.right_hand_id)[0]
        self.left_forearm_vertices = np.where(self.smpl_index == self.left_forearm_id)[0]
        self.right_forearm_vertices = np.where(self.smpl_index == self.right_forearm_id)[0]

    def __call__(self, **kwargs):
        """
        Phiên bản advanced: thu nhỏ bàn tay thay vì loại bỏ hoàn toàn
        """
        try:
            # Lấy body_model và params từ kwargs
            body_model = kwargs['body_model']
            params = kwargs['params']
            
            # Tính toán output từ body_model
            output = body_model(**params)
            
            # output là tensor với shape [batch_size, num_vertices, 3]
            vertices = output.clone()
            batch_size = vertices.shape[0]
            
            # Xử lý từng batch
            for b in range(batch_size):
                # Thu nhỏ bàn tay trái
                if len(self.left_hand_vertices) > 0:
                    left_hand_coords = vertices[b, self.left_hand_vertices, :]
                    left_hand_center = left_hand_coords.mean(dim=0, keepdim=True)
                    left_hand_shrinked = left_hand_center + (left_hand_coords - left_hand_center) * self.shrink_factor
                    vertices[b, self.left_hand_vertices, :] = left_hand_shrinked
                
                # Thu nhỏ bàn tay phải
                if len(self.right_hand_vertices) > 0:
                    right_hand_coords = vertices[b, self.right_hand_vertices, :]
                    right_hand_center = right_hand_coords.mean(dim=0, keepdim=True)
                    right_hand_shrinked = right_hand_center + (right_hand_coords - right_hand_center) * self.shrink_factor
                    vertices[b, self.right_hand_vertices, :] = right_hand_shrinked
            
            # Trả về dictionary với key 'vertices'
            return {'vertices': vertices}
            
        except Exception as e:
            print(f"Error in RemoveHandsAdvanced: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback
            if 'body_model' in kwargs and 'params' in kwargs:
                body_model = kwargs['body_model']
                params = kwargs['params']
                output = body_model(**params)
                return {'vertices': output}
            else:
                raise e