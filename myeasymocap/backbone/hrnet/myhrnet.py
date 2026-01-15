import os
import numpy as np
import math
import cv2
import torch
from ..basetopdown import BaseTopDownModelCache
from .hrnet import HRNet

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim: {}'.format(batch_heatmaps.shape)

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

COCO17_IN_BODY25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
pairs = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]

def coco17tobody25(points2d):
    kpts = np.zeros((points2d.shape[0], 25, 3))
    kpts[:, COCO17_IN_BODY25, :2] = points2d[:, :, :2]
    kpts[:, COCO17_IN_BODY25, 2:3] = points2d[:, :, 2:3]
    kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
    kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
    kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
    kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)
    return kpts

class MyHRNet(BaseTopDownModelCache):
    def __init__(self, ckpt, single_person=True, num_joints=17, name='keypoints2d', 
                 test_multiscale=False, multiscale_factors=[0.8, 1.0, 1.2]):
        super().__init__(name, bbox_scale=1.25, res_input=[288, 384])
        # Nếu bật multi-scale testing
        self.test_multiscale = test_multiscale
        self.multiscale_factors = multiscale_factors
        
        # Nếu启用，那么將每個視角最多保留一個，並且squeeze and stack
        self.single_person = single_person
        model = HRNet(48, num_joints, 0.1)
        self.num_joints = num_joints
        
        if not os.path.exists(ckpt) and ckpt.endswith('pose_hrnet_w48_384x288.pth'):
            url = "11ezQ6a_MxIRtj26WqhH3V3-xPI3XqYAw"
            text = '''Download `models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth` from (OneDrive)[https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ],
            And place it into {}'''.format(os.path.dirname(ckpt))
            print(text)
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            cmd = 'gdown "{}" -O {}'.format(url, ckpt)
            print('\n', cmd, '\n')
            os.system(cmd)
        assert os.path.exists(ckpt), f'{ckpt} not exists'
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        self.model = model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        
        if self.test_multiscale:
            print(f"[MyHRNet] Multi-scale testing ENABLED with scales: {self.multiscale_factors}")

    @staticmethod
    def get_max_preds(batch_heatmaps):
        coords, maxvals = get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        # post-processing
        if True:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = batch_heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                        diff = np.array(
                            [
                                hm[py][px+1] - hm[py][px-1],
                                hm[py+1][px]-hm[py-1][px]
                            ]
                        )
                        coords[n][p] += np.sign(diff) * .25
        coords = coords.astype(np.float32) * 4
        pred = np.dstack((coords, maxvals))
        return pred

    def process_single_scale(self, bbox, img, imgname, scale=1.0):
        """
        Xử lý ảnh ở một scale cụ thể
        """
        # Tạo bbox mới với scale
        scaled_bbox = bbox.copy()
        if scale != 1.0:
            # Tính center và size của bbox
            center_x = (scaled_bbox[:, 0] + scaled_bbox[:, 2]) / 2
            center_y = (scaled_bbox[:, 1] + scaled_bbox[:, 3]) / 2
            width = scaled_bbox[:, 2] - scaled_bbox[:, 0]
            height = scaled_bbox[:, 3] - scaled_bbox[:, 1]
            
            # Scale size
            width = width * scale
            height = height * scale
            
            # Tạo bbox mới
            scaled_bbox[:, 0] = center_x - width / 2
            scaled_bbox[:, 1] = center_y - height / 2
            scaled_bbox[:, 2] = center_x + width / 2
            scaled_bbox[:, 3] = center_y + height / 2
        
        # Gọi hàm parent để xử lý
        out = super(MyHRNet, self).__call__(scaled_bbox, img, imgname)
        return out['params']['output'], out['params']['inv_trans']

    def aggregate_multiscale_heatmaps(self, heatmaps_list):
        """
        Tổng hợp heatmaps từ nhiều scales
        Args:
            heatmaps_list: List of (heatmap, inv_trans) tuples
        Returns:
            aggregated_heatmap: Heatmap trung bình
            inv_trans: Transform từ scale gốc (1.0)
        """
        # Lấy kích thước từ scale gốc (giả sử scale=1.0 là scale giữa)
        base_idx = len(heatmaps_list) // 2
        base_shape = heatmaps_list[base_idx][0].shape
        
        aggregated = np.zeros(base_shape, dtype=np.float32)
        
        for heatmap, _ in heatmaps_list:
            # Resize về kích thước base nếu cần
            if heatmap.shape != base_shape:
                resized_heatmap = np.zeros(base_shape, dtype=np.float32)
                for b in range(heatmap.shape[0]):
                    for j in range(heatmap.shape[1]):
                        resized_heatmap[b, j] = cv2.resize(
                            heatmap[b, j], 
                            (base_shape[3], base_shape[2]),
                            interpolation=cv2.INTER_LINEAR
                        )
                aggregated += resized_heatmap
            else:
                aggregated += heatmap
        
        # Tính trung bình
        aggregated /= len(heatmaps_list)
        
        # Trả về heatmap tổng hợp và inv_trans từ scale gốc
        return aggregated, heatmaps_list[base_idx][1]

    def __call__(self, bbox, images, imgnames):
        squeeze = False
        if not isinstance(images, list):
            images = [images]
            imgnames = [imgnames]
            bbox = [bbox]
            squeeze = True
        nViews = len(images)
        kpts_all = []
        
        for nv in range(nViews):
            _bbox = bbox[nv]
            if _bbox.shape[0] == 0:
                if self.single_person:
                    kpts = np.zeros((1, self.num_joints, 3))
                else:
                    kpts = np.zeros((_bbox.shape[0], self.num_joints, 3))
            else:
                img = images[nv]
                
                if self.test_multiscale:
                    # Multi-scale testing
                    heatmaps_list = []
                    for scale in self.multiscale_factors:
                        output, inv_trans = self.process_single_scale(_bbox, img, imgnames[nv], scale)
                        heatmaps_list.append((output, inv_trans))
                    
                    # Tổng hợp heatmaps
                    aggregated_heatmap, inv_trans = self.aggregate_multiscale_heatmaps(heatmaps_list)
                    
                    # Trích xuất keypoints từ heatmap tổng hợp
                    kpts = self.get_max_preds(aggregated_heatmap)
                    kpts_ori = self.batch_affine_transform(kpts, inv_trans)
                    kpts = np.concatenate([kpts_ori, kpts[..., -1:]], axis=-1)
                else:
                    # Single-scale testing (logic gốc)
                    out = super().__call__(_bbox, img, imgnames[nv])
                    output = out['params']['output']
                    kpts = self.get_max_preds(output)
                    kpts_ori = self.batch_affine_transform(kpts, out['params']['inv_trans'])
                    kpts = np.concatenate([kpts_ori, kpts[..., -1:]], axis=-1)
            
            kpts = coco17tobody25(kpts)
            kpts_all.append(kpts)
        
        if self.single_person:
            kpts_all = [k[0] for k in kpts_all]
            kpts_all = np.stack(kpts_all)
        if squeeze:
            kpts_all = kpts_all[0]
        
        return {
            'keypoints': kpts_all
        }