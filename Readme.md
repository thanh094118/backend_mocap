# ============================================
# FILE 6: README_MODELS.md
# ============================================
# Model Files Setup

## Models Downloaded from Google Drive

During Docker build, the following models are automatically downloaded:

1. **HRNet Pose Model** (`data/models/pose_hrnet_w48_384x288.pth`)
   - Size: ~200MB
   - Source: Google Drive

2. **PARE Checkpoint** (`models/pare/data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt`)
   - Size: ~500MB
   - Source: Google Drive

3. **SMPL Neutral Model** (`models/pare/data/body_models/smpl/SMPL_NEUTRAL.pkl`)
   - Size: ~100MB
   - Source: Google Drive

4. **YOLOv5 Model** (`models/yolo/yolov5m.pt`)
   - Size: ~40MB
   - Source: Google Drive

## Manual Download (if needed)

If automatic download fails, manually download:

```bash
# Create directories
mkdir -p data/models
mkdir -p models/pare/data/pare/checkpoints
mkdir -p models/pare/data/body_models/smpl
mkdir -p models/yolo

# Download using gdown
pip install gdown

gdown 1eZPkFzRN_TL_tUvfRTofiqWngproIirZ -O data/models/pose_hrnet_w48_384x288.pth
gdown 1SRrH_ha122KD4z_PjJ0UDm_4U1ti97X- -O models/pare/data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt
gdown 1Rza5kVxB7Lp5lP_o0r3LCpmk1LaepiI- -O models/pare/data/body_models/smpl/SMPL_NEUTRAL.pkl
gdown 1bv56ZN7tRIoPXPfeow26rBdC1NddLdS2 -O models/yolo/yolov5m.pt
```

## Why Not Commit Models to Git?

- **Size**: Total ~840MB - exceeds GitHub's limits
- **Storage**: Uses Git LFS quota
- **Speed**: Slow clone/pull times
- **Best Practice**: Store large files in cloud storage

## Alternative: Use Cloud Storage

For production, consider:
- **Google Cloud Storage**
- **AWS S3**
- **Hugging Face Hub**
