import os
import sys
import torch
import torch.nn as nn
import numpy as np
from os.path import join
import cv2

from ..basetopdown import BaseTopDownModelCache

# CLIFF paths - cần clone repo thủ công
CLIFF_ROOT = 'third_party/CLIFF'
CLIFF_CKPT = 'data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt'

class MyCLIFF(BaseTopDownModelCache):
    """
    CLIFF wrapper cho EasyMocap
    
    Setup:
    1. Clone CLIFF repo:
       git clone https://github.com/huawei-noah/noah-research.git
       cp -r noah-research/CLIFF third_party/CLIFF
    
    2. Download checkpoint:
       wget https://drive.google.com/uc?id=1SKKdHF4FKXflnc2dHjpVmEV7CVQYZ68R
       
    3. Install dependencies:
       cd third_party/CLIFF
       pip install -r requirements.txt
    """
    
    def __init__(self, ckpt='hr48', use_output_avg=False, num_runs=3, 
                 focal_length=5000., img_size=224) -> None:
        """
        Args:
            ckpt: checkpoint path hoặc 'hr48'/'res50'
            use_output_avg: có sử dụng output averaging không
            num_runs: số lần chạy inference khi dùng output averaging
            focal_length: tiêu cự camera mặc định
            img_size: kích thước input image (CLIFF dùng 224)
        """
        super().__init__('cliff', bbox_scale=1.25, res_input=img_size)
        
        self.use_output_avg = use_output_avg
        self.num_runs = num_runs
        self.focal_length = focal_length
        self.img_size = img_size
        
        # Add CLIFF to path
        if not os.path.exists(CLIFF_ROOT):
            raise FileNotFoundError(
                f"CLIFF not found at {CLIFF_ROOT}\n"
                "Please clone: git clone https://github.com/huawei-noah/noah-research.git\n"
                "Then: cp -r noah-research/CLIFF third_party/CLIFF"
            )
        
        sys.path.insert(0, CLIFF_ROOT)
        
        # Setup device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Load model
        self.model = self._build_cliff_model(ckpt)
        self.model.eval()
        self.model.to(self.device)
        
        if self.use_output_avg:
            print(f"[CLIFF] Output Averaging ENABLED with {self.num_runs} runs")
        
        print(f"[CLIFF] Model loaded on {self.device}")
    
    def _build_cliff_model(self, ckpt_name):
        """Build CLIFF model từ source code"""
        try:
            # Import từ CLIFF source
            # CLIFF dùng HRNet backbone
            import common.config as config
            from models import cliff
            
            # Load config
            cfg = config.get_cfg_defaults()
            
            # Tìm checkpoint path
            if os.path.exists(ckpt_name):
                ckpt_path = ckpt_name
            elif ckpt_name == 'hr48':
                ckpt_path = CLIFF_CKPT
            else:
                ckpt_path = ckpt_name
            
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {ckpt_path}\n"
                    "Download from: https://drive.google.com/uc?id=1SKKdHF4FKXflnc2dHjpVmEV7CVQYZ68R"
                )
            
            # Build model
            model = cliff.CLIFF(
                constants.SMPL_MEAN_PARAMS,
                focal_length=self.focal_length,
                img_res=self.img_size
            )
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if exists
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            print(f"[CLIFF] Loaded checkpoint from {ckpt_path}")
            
            return model
            
        except ImportError as e:
            raise ImportError(
                f"Cannot import CLIFF modules: {e}\n"
                "Make sure CLIFF is properly set up:\n"
                "1. Clone repo and copy to third_party/CLIFF\n"
                "2. Install dependencies: pip install -r requirements.txt"
            )
    
    def __call__(self, bbox, images, imgnames):
        """
        Forward inference
        
        Args:
            bbox: list of bounding boxes [[x, y, w, h]]
            images: input image (numpy array)
            imgnames: image filenames
            
        Returns:
            dict: {
                'params': {
                    'Rh': global rotation [1, 3],
                    'Th': translation [1, 3],
                    'poses': body poses [1, 69],
                    'shapes': shape parameters [1, 10]
                }
            }
        """
        if not self.use_output_avg:
            return self._single_inference(bbox[0], images, imgnames)
        
        # Output Averaging
        outputs = []
        for _ in range(self.num_runs):
            out = self._single_inference(bbox[0], images, imgnames)
            
            output_detached = {}
            for k, v in out['params'].items():
                if isinstance(v, torch.Tensor):
                    output_detached[k] = v.detach().cpu()
                else:
                    output_detached[k] = v
            
            outputs.append(output_detached)
        
        # Average results
        avg_results = {}
        for k in outputs[0].keys():
            if isinstance(outputs[0][k], torch.Tensor):
                stacked = torch.stack([o[k] for o in outputs])
                avg_results[k] = stacked.mean(dim=0).to(self.device)
            else:
                avg_results[k] = outputs[0][k]
        
        return {'params': avg_results}
    
    def _single_inference(self, bbox, image, imgname):
        """Single forward pass"""
        with torch.no_grad():
            # Crop image
            img_crop, norm_img, center, scale = self._crop_and_normalize(image, bbox)
            
            # Prepare batch
            batch = {
                'img': norm_img.unsqueeze(0).to(self.device),
                'img_h': torch.FloatTensor([image.shape[0]]).to(self.device),
                'img_w': torch.FloatTensor([image.shape[1]]).to(self.device),
                'focal_length': torch.FloatTensor([self.focal_length]).to(self.device),
                'center': torch.FloatTensor(center).unsqueeze(0).to(self.device),
                'scale': torch.FloatTensor([scale]).to(self.device),
            }
            
            # Forward pass
            output = self.model(batch)
            
            # Parse output
            params = self._parse_output(output, batch)
            
            return {'params': params}
    
    def _crop_and_normalize(self, image, bbox):
        """Crop và normalize image theo CLIFF preprocessing"""
        x, y, w, h = bbox
        
        # Calculate center and scale
        center = np.array([x + w/2, y + h/2], dtype=np.float32)
        scale = max(w, h) * self.bbox_scale / 200.0  # CLIFF scale format
        
        # Crop and resize using CLIFF's crop function
        # (simplified version - thực tế cần dùng hàm crop_img từ CLIFF)
        box_size = int(scale * 200)
        x1 = int(max(0, center[0] - box_size/2))
        y1 = int(max(0, center[1] - box_size/2))
        x2 = int(min(image.shape[1], center[0] + box_size/2))
        y2 = int(min(image.shape[0], center[1] + box_size/2))
        
        img_crop = image[y1:y2, x1:x2]
        img_crop = cv2.resize(img_crop, (self.img_size, self.img_size))
        
        # Normalize theo ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        norm_img = img_crop.astype(np.float32) / 255.0
        norm_img = (norm_img - mean) / std
        
        # Convert to tensor
        norm_img = torch.from_numpy(norm_img).float()
        norm_img = norm_img.permute(2, 0, 1)  # [3, H, W]
        
        return img_crop, norm_img, center, scale
    
    def _parse_output(self, output, batch):
        """Parse CLIFF output sang EasyMocap format"""
        # CLIFF output
        pred_rotmat = output['pred_pose']      # [B, 24, 3, 3]
        pred_betas = output['pred_shape']       # [B, 10]
        pred_cam = output['pred_cam']           # [B, 3]
        
        # Convert rotation matrix to axis-angle
        batch_size = pred_rotmat.shape[0]
        rotmat_flat = pred_rotmat.reshape(-1, 3, 3)
        
        # Sử dụng rotation_matrix_to_angle_axis từ PyTorch3D hoặc tự implement
        rvec_flat = self._rotmat_to_axisangle(rotmat_flat)
        rvec = rvec_flat.reshape(batch_size, 24, 3)
        
        # Flatten poses
        poses_flat = rvec.reshape(batch_size, -1)
        
        # Parse camera để tính translation
        # CLIFF: pred_cam = [scale, tx, ty]
        cam_scale = pred_cam[:, [0]]  # [B, 1]
        cam_trans = pred_cam[:, 1:]   # [B, 2]
        
        # Convert weak-perspective to 3D translation
        focal_length = batch['focal_length']
        img_size = self.img_size
        
        # Depth from scale
        # CLIFF uses: s = f / (depth * img_size)
        depth = focal_length / (cam_scale * img_size + 1e-9)
        
        # Full 3D translation
        tx = cam_trans[:, [0]] * depth
        ty = cam_trans[:, [1]] * depth
        tz = depth
        
        translation = torch.cat([tx, ty, tz], dim=-1)
        
        params = {
            'Rh': poses_flat[:, :3],      # Global rotation [B, 3]
            'Th': translation,             # Translation [B, 3]
            'poses': poses_flat[:, 3:],   # Body poses [B, 69]
            'shapes': pred_betas,          # Shape [B, 10]
        }
        
        return params
    
    def _rotmat_to_axisangle(self, rotmat):
        """Convert rotation matrix to axis-angle representation"""
        # Simplified conversion (trong thực tế nên dùng PyTorch3D)
        batch_size = rotmat.shape[0]
        
        # Use Rodrigues formula
        trace = rotmat[:, 0, 0] + rotmat[:, 1, 1] + rotmat[:, 2, 2]
        theta = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        
        # Axis
        r = torch.stack([
            rotmat[:, 2, 1] - rotmat[:, 1, 2],
            rotmat[:, 0, 2] - rotmat[:, 2, 0],
            rotmat[:, 1, 0] - rotmat[:, 0, 1]
        ], dim=-1)
        
        r_norm = torch.norm(r, dim=-1, keepdim=True) + 1e-8
        axis = r / r_norm
        
        # Angle-axis
        angle_axis = axis * theta.unsqueeze(-1)
        
        return angle_axis


# Helper function để setup CLIFF
def setup_cliff():
    """
    Hướng dẫn setup CLIFF đúng cách
    """
    print("""
    === CLIFF Setup Guide ===
    
    1. Clone CLIFF repository:
       git clone https://github.com/huawei-noah/noah-research.git
       mkdir -p third_party
       cp -r noah-research/CLIFF third_party/CLIFF
    
    2. Install dependencies:
       cd third_party/CLIFF
       pip install -r requirements.txt
       pip install torch torchvision
    
    3. Download checkpoint (HR48):
       mkdir -p data/ckpt
       # Download từ Google Drive:
       # https://drive.google.com/uc?id=1SKKdHF4FKXflnc2dHjpVmEV7CVQYZ68R
       # Hoặc dùng gdown:
       pip install gdown
       gdown 1SKKdHF4FKXflnc2dHjpVmEV7CVQYZ68R -O data/ckpt/cliff_hr48.pt
    
    4. Download SMPL model:
       # Register tại https://smpl.is.tue.mpg.de/
       # Download SMPL_python_v.1.0.0.zip
       # Extract và copy basicModel_neutral_lbs_10_207_0_v1.0.0.pkl vào:
       mkdir -p data/smpl
       cp basicModel_neutral_lbs_10_207_0_v1.0.0.pkl data/smpl/
    
    5. Test installation:
       cd third_party/CLIFF
       python demo.py --img_file demo.png
    """)


if __name__ == '__main__':
    setup_cliff()