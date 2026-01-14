from tqdm import tqdm
import cv2
import os
from easymocap.visualize.pyrender_wrapper import plot_meshes
from os.path import join
import numpy as np
from easymocap.datasets.base import add_logo
from easymocap.mytools.vis_base import merge, plot_bbox
from easymocap.mytools.camera_utils import Undistort
from .vis import VisBase
import pickle

class Render(VisBase):
    def __init__(self, name='render', scale=0.5, backend='pyrender',
        part_segmentation_path='models/smpl_partSegmentation_mapping.pkl',
                 **kwargs) -> None:
        super().__init__(name=name, scale=1., **kwargs)
        self.scale3d = scale
        
        # Load part segmentation data
        self.part_mapping = self._load_part_segmentation(part_segmentation_path)
        
        # Part IDs for each limb (each limb gets unique color)
        self.right_arm_ids = [17, 19]       # R_Shoulder, R_Elbow
        self.left_arm_ids = [16, 18]        # L_Shoulder, L_Elbow
        self.right_leg_ids = [2, 5]         # R_Hip, R_Knee
        self.left_leg_ids = [1, 4]          # L_Hip, L_Knee
        
        # Get indices for each limb
        self.right_arm_indices = self._get_multiple_part_indices(self.right_arm_ids)
        self.left_arm_indices = self._get_multiple_part_indices(self.left_arm_ids)
        self.right_leg_indices = self._get_multiple_part_indices(self.right_leg_ids)
        self.left_leg_indices = self._get_multiple_part_indices(self.left_leg_ids)
        
        # User choices
        self.remove_hands = None
        self.use_colors = None  # NEW: User choice for coloring
        self.use_lighting = None
        
        self.remove_ids = [20, 21, 22, 23]  # L_Hand, R_Hand, Jaw, L_Eye
        self.remove_indices = self._get_multiple_part_indices(self.remove_ids)

    def _load_part_segmentation(self, path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            if 'smpl_index' in data:
                part_mapping = data['smpl_index']
                return part_mapping
            else:
                return None
        except Exception as e:
            print(f"✗ Error loading part segmentation: {e}")
            return None

    def _get_multiple_part_indices(self, part_ids):
        if self.part_mapping is None:
            return np.array([])

        if isinstance(self.part_mapping, (list, np.ndarray)):
            indices = []
            for pid in part_ids:
                idx = np.where(np.array(self.part_mapping) == pid)[0]
                indices.extend(idx.tolist())
            return np.array(indices)
        else:
            return np.array([])
    def _create_vertex_colors(self, num_vertices):
        """
        part ID:
        - ID 1 (L_Hip): Magenta (1.0, 0.0, 1.0)
        - ID 2 (R_Hip): Cyan (0.0, 1.0, 1.0)
        - ID 4 (L_Knee): Orange (1.0, 0.65, 0.0)
        - ID 5 (R_Knee): Purple (0.628, 0.125, 0.949)
        - ID 16 (L_Shoulder): Lime (0.0, 1.0, 0.0)
        - ID 17 (R_Shoulder): Red (1.0, 0.0, 0.0)
        - ID 18 (L_Elbow): Navy (0.0, 0.0, 0.502)
        - ID 19 (R_Elbow): Coral (1.0, 0.498, 0.314)
        - Body (other): Neutral beige (0.8, 0.75, 0.7)
        """
        # Default color: neutral beige for body
        colors = np.ones((num_vertices, 3)) * np.array([0.8, 0.75, 0.7])
        
        # Color mapping for each part ID
        color_map = {
            1: np.array([1.0, 0.0, 1.0]),      # L_Hip: Magenta
            2: np.array([0.0, 1.0, 1.0]),      # R_Hip: Cyan
            4: np.array([1.0, 0.0, 1.0]),     # L_Knee: option-Orange: [1.0, 0.65, 0.0]
            5: np.array([0.0, 1.0, 1.0]),      # R_Knee: option-Red [0.0, 0.0, 1.0]
            16: np.array([1.0, 1.0, 0.0]),     # L_Shoulder: Lime
            17: np.array([0.0, 0.0, 1.0]),     # R_Shoulder: Purple
            18: np.array([1.0, 1.0, 0.0]),   # L_Elbow: Navy [0.0, 0.0, 0.502]
            19: np.array([0.0, 0.0, 1.0])  # R_Elbow: Coral [1.0, 0.498, 0.314]0.0, 0.0, 1.0
        }
        
        if self.part_mapping is not None:
            # Iterate through each part ID and assign colors
            for part_id, color in color_map.items():
                indices = np.where(np.array(self.part_mapping) == part_id)[0]
                if len(indices) > 0:
                    colors[indices] = color
        
        return colors

    def _remove_parts_mesh(self, vertices, faces, vertex_colors=None):
        """Remove vertices and faces belonging to specified parts"""
        remove_indices = set(self.remove_indices.tolist())
        keep_mask = np.array([i not in remove_indices for i in range(len(vertices))])

        # Create mapping from old to new indices
        old_to_new = {}
        new_idx = 0
        for old_idx, keep in enumerate(keep_mask):
            if keep:
                old_to_new[old_idx] = new_idx
                new_idx += 1

        # Filter vertices
        new_vertices = vertices[keep_mask]
        
        # Filter vertex colors if provided
        new_colors = None
        if vertex_colors is not None:
            new_colors = vertex_colors[keep_mask]

        # Filter faces
        keep_faces = []
        for f in faces:
            if all(v in old_to_new for v in f):
                keep_faces.append([old_to_new[v] for v in f])
        new_faces = np.array(keep_faces, dtype=np.int32)

        return new_vertices, new_faces, new_colors

    def _get_user_choice_remove_hands(self):
        if self.remove_hands is None:
            while True:
                choice = input("Remove hands (y/n)? ").lower().strip()
                if choice in ['y', 'yes']:
                    self.remove_hands = True
                    print("Hands will be removed from rendering")
                    break
                elif choice in ['n', 'no']:
                    self.remove_hands = False
                    print("Hands will be kept in rendering")
                    break
        return self.remove_hands

    def _get_user_choice_colors(self):
        if self.use_colors is None:
            while True:
                choice = input("Use colors for limbs (y/n)? ").lower().strip()
                if choice in ['y', 'yes']:
                    self.use_colors = True
                    print("Limbs will be colored")
                    break
                elif choice in ['n', 'no']:
                    self.use_colors = False
                    print("Default (no special colors)")
                    break
        return self.use_colors

    def _get_user_choice_lighting(self):
        """Hỏi user có muốn bật lighting không"""
        if not hasattr(self, 'use_lighting') or self.use_lighting is None:
            while True:
                choice = input("Enable lighting/shading (y/n)? ").lower().strip()
                if choice in ['y', 'yes']:
                    self.use_lighting = True
                    print("✓ Lighting enabled (3D look with shadows)")
                    break
                elif choice in ['n', 'no']:
                    self.use_lighting = False
                    print("✓ Lighting disabled (flat colors)")
                    break
        return self.use_lighting
    
    def __call__(self, body_model, params, cameras, imgnames):
        """
        Render mesh for each image in imgnames.
        
        With optional colored limbs:
        - Right arm: RED
        - Left arm: GREEN
        - Right leg: BLUE
        - Left leg: YELLOW
        """
        # Get user choices at the beginning
        remove_hands = self._get_user_choice_remove_hands()
        use_colors = self._get_user_choice_colors()
        use_lighting = self._get_user_choice_lighting()
        
        vertices = body_model.vertices(params, return_tensor=False)
        faces = body_model.faces
        
        for nf, img in enumerate(tqdm(imgnames, desc=self.name)):
            basename = os.path.basename(img)
            assert os.path.exists(img), img
            
            vis = cv2.imread(img)
            vis = cv2.resize(vis, None, fx=self.scale3d, fy=self.scale3d)
            
            vert = vertices[nf]
            
            # Create vertex colors if user wants them
            vertex_colors = None
            if use_colors and self.part_mapping is not None:
                vertex_colors = self._create_vertex_colors(len(vert))
            
            # Apply hand removal if requested
            if remove_hands and self.part_mapping is not None:
                filtered_vert, filtered_faces, filtered_colors = self._remove_parts_mesh(
                    vert, faces, vertex_colors
                )
            else:
                filtered_vert, filtered_faces = vert, faces
                filtered_colors = vertex_colors
            
            # Build mesh dictionary
            meshes = {
                0: {
                    'vertices': filtered_vert,
                    'faces': filtered_faces,
                    'id': 0,
                    'name': f'human_{0}',
                    'smooth': False if use_colors else True
                }
            }
            
            # Add colors only if user wants them
            if use_colors and filtered_colors is not None:
                meshes[0]['colors'] = filtered_colors
            
            K = cameras['K'][nf].copy()
            K[:2, :] *= self.scale3d
            R = cameras['R'][nf]
            T = cameras['T'][nf]
            
            ret = plot_meshes(vis, meshes, K, R, T, mode='image', use_lighting=use_lighting)
            self.merge_and_write([ret])

class Render_multiview(VisBase):
    def __init__(self, view_list=[], name='vis_render', model_name='body_model', render_mode='image', backend='pyrender', shape=[-1,-1], scale=1., **kwargs):
        self.scale3d = scale
        super().__init__(name=name, scale=1., **kwargs)
        self.view_list = view_list
        self.render_mode = render_mode
        self.model_name = model_name
        self.shape = shape

    def render_frame(self, imgname, vert, faces, cameras, pids=[]):
        mv_ret = []
        if not isinstance(imgname, list):
            imgname = [imgname]
        for nv in self.view_list:
            basename = os.path.basename(imgname[nv])
            assert os.path.exists(imgname[nv]), imgname[nv]
            vis = cv2.imread(imgname[nv])
            # undistort the images
            if cameras['dist'] is not None:
                vis = Undistort.image(vis, cameras['K'][nv], cameras['dist'][nv], sub=os.path.basename(os.path.dirname(imgname[nv])))
            vis = cv2.resize(vis, None, fx=self.scale3d, fy=self.scale3d)
            meshes = {}
            if vert.ndim == 2:
                meshes[0] = {
                    'vertices': vert,
                    'faces': faces,
                    'id': 0,
                    'name': 'human_{}'.format(0)
                }
            elif vert.ndim == 3:
                if len(pids) == 0:
                    pids = list(range(vert.shape[0]))
                for ipid, pid in enumerate(pids):
                    meshes[pid] = {
                        'vertices': vert[ipid],
                        'faces': faces,
                        'id': pid,
                        'name': 'human_{}'.format(pid)
                    }
            K = cameras['K'][nv].copy()
            K[:2, :] *= self.scale3d
            R = cameras['R'][nv]
            T = cameras['T'][nv]
            # add ground
            if self.render_mode == 'ground':
                from easymocap.visualize.geometry import create_ground
                ground = create_ground(
                    center=[0, 0, -0.05], xdir=[1, 0, 0], ydir=[0, 1, 0], # 位置
                    step=1, xrange=10, yrange=10, # 尺寸
                    white=[1., 1., 1.], black=[0.5,0.5,0.5], # 颜色
                    two_sides=True
                )
                meshes[1001] = ground
                vis = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8) + 255
                focal = min(self.shape) * 1.2
                K = np.array([
                    [focal,0,vis.shape[0]/2],
                    [0,focal,vis.shape[1]/2],
                    [0,0,1]])
                ret = plot_meshes(vis, meshes, K, R, T, mode='rgb')
            else:
                ret = plot_meshes(vis, meshes, K, R, T, mode=self.render_mode)
            mv_ret.append(ret)
        self.merge_and_write(mv_ret)

    def render_(self, vertices, faces, cameras, imgnames, pids=[]):
        for nf, imgname in enumerate(tqdm(imgnames, desc=self.name)):
            vert = vertices[nf]
            camera_ = {cam: val[nf] for cam, val in cameras.items()}
            self.render_frame(imgname, vert, faces, camera_, pids=pids)

    def __call__(self, params, cameras, imgnames, **kwargs):
        body_model = kwargs[self.model_name]
        vertices = body_model.vertices(params, return_tensor=False)
        faces = body_model.faces
        self.render_(vertices, faces, cameras, imgnames)

class RenderAll_multiview(Render_multiview):
    def __call__(self, results, cameras, imgnames, meta, **kwargs):
        body_model = kwargs[self.model_name]
        for index in tqdm(meta['index'], desc=self.name):
            results_frame = []
            for pid, result in results.items():
                if index >= result['frames'][0] and index <= result['frames'][-1]:
                    frame_rel = result['frames'].index(index)
                    results_frame.append({
                        'id': pid,
                    })
                    for key in ['Rh', 'Th', 'poses', 'shapes']:
                        if result['params'][key].shape[0] == 1:
                            results_frame[-1][key] = result['params'][key]
                        else:
                            results_frame[-1][key] = result['params'][key][frame_rel:frame_rel+1]
            params = {}
            for key in results_frame[0].keys():
                if key != 'id':
                    params[key] = np.concatenate([res[key] for res in results_frame], axis=0)
            pids = [res['id'] for res in results_frame]
            vertices = body_model.vertices(params, return_tensor=False)
            camera_ = {cam: val[index] for cam, val in cameras.items()}
            self.render_frame(imgnames[index], vertices, body_model.faces, camera_, pids=pids)
            # self.render_frame(vertices, body_model.faces, camera_, imgnames[index], pids=pids)
            # self.render_frame(vertices, body_model.faces, camera_, imgnames[index], pids=pids)
    
class Render_nocam:
    def __init__(self, scale=0.5, backend='pyrender',view_list=[0]) -> None:
        self.name = 'render'
        self.scale = scale
        self.view_list = view_list

    def __call__(self, hand_model, params, images):

        vertices = hand_model(**params, return_verts=True, return_tensor=False)
        faces = hand_model.faces
        for nf, img in enumerate(tqdm(images, desc=self.name)):
            for nv in self.view_list:
                if isinstance(img, np.ndarray):
                    vis = img.copy()
                    basename = '{:06}.jpg'.format(nf)
                else:
                    basename = os.path.basename(img[nv])
                    # 重新读入图片
                    assert os.path.exists(img[nv]), img[nv]
                    vis = cv2.imread(img[nv])
                    
                vis = cv2.resize(vis, None, fx=self.scale, fy=self.scale)
                vert = vertices[nf]
                meshes = {}
                meshes[0] = {
                    'vertices': vert,
                    'faces': faces,
                    'id': 0,
                    'name': 'human_{}'.format(0)
                }
                K = np.array([[vis.shape[0],0,vis.shape[0]/2],[0,vis.shape[1],vis.shape[1]/2],[0,0,1]])
                K[:2, :] *= self.scale
                R = np.eye(3)
                T = np.array([0,0,0.3])
                ret = plot_meshes(vis, meshes, K, R, T, mode='image')
                outname = join(self.output, self.name, basename)
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                cv2.imwrite(outname, ret)

class Render_multiview_hand(Render_multiview):
    def __call__(self, hand_model_l, params_l, cameras, imgnames):
        vertices = hand_model_l(**params_l, return_verts=True, return_tensor=False)
        faces = hand_model_l.faces
        self.render_(vertices, faces, cameras, imgnames)
        
class Render_smplh(Render_multiview):
    def __init__(self, path, at_step, scale=0.5, mode='image', backend='pyrender', view_list=[0]) -> None:
        super().__init__(scale, mode, backend, view_list)
        from easymocap.config import Config, load_object
        cfg_data = Config.load(path)
        self.model = load_object(cfg_data.module, cfg_data.args)
        self.at_step = at_step

    def __call__(self, params_smplh, cameras, imgnames):
        vertices = self.model(return_verts=True, return_tensor=False, **params_smplh)
        faces = self.model.faces
        if self.at_step:
            self.render_([vertices], faces, cameras, [imgnames])
        else:
            self.render_(vertices, faces, cameras, imgnames)

class Render_smplh2(Render_smplh):
    def __call__(self, params, cameras, imgnames):
        super().__call__(params, cameras, imgnames)

def projectPoints(X, K, R, t, Kd):    
    x = R @ X + t
    x[0:2,:] = x[0:2,:]/x[2,:]#到归一化平面
    r = x[0,:]*x[0,:] + x[1,:]*x[1,:]

    x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
    x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])
    x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
    x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
    return x
class Render_multiview_handbyk3d(Render_multiview):
    def __call__(self, hand_model_l, params_l, hand_model_r, params_r, cameras, imgnames, keypoints3d):
        # breakpoint()
        joint_regressor_r = np.load('models/handmesh/data/joint_regressor_r.npy') #右手
        joint_regressor_l = np.load('models/handmesh/data/joint_regressor_l.npy') #右手
        facesl = hand_model_l.faces
        facesr = hand_model_r.faces

        # for nf, img in enumerate(tqdm(imgnames, desc=self.name)):
        #不显示0号人物的结果
        keypoints3d[0]=0

        img = imgnames
        k3d = keypoints3d
        
        vertices_l = hand_model_l(**params_l, return_verts=True, return_tensor=False) #[nf]
        vertices_r = hand_model_r(**params_r, return_verts=True, return_tensor=False) #[nf]

        # breakpoint()

        joint_l = np.repeat(joint_regressor_l[None, :, :],vertices_l.shape[0],0) @ vertices_l
        joint_r = np.repeat(joint_regressor_r[None, :, :],vertices_r.shape[0],0) @ vertices_r
        params_l['Th']+=k3d[:,7,:3] - joint_l[:,0,:] #左手7右手4 #[nf]
        params_r['Th']+=k3d[:,4,:3] - joint_r[:,0,:] #左手7右手4 #[nf]
        vertices_l = hand_model_l(**params_l, return_verts=True, return_tensor=False) #[nf]
        vertices_r = hand_model_r(**params_r, return_verts=True, return_tensor=False) #[nf]

        faces = []
        vert = []
        pids = []
        for i in range(k3d.shape[0]):
            if k3d[i,7,-1]==0:
                continue
            vv = vertices_l[i].copy()
            vert.append(vv)
            faces.append(facesl)
            pids.append(i)
        
        for i in range(k3d.shape[0]):
            if k3d[i,4,-1]==0:
                continue
            vv = vertices_r[i].copy()
            vert.append(vv)
            faces.append(facesr)
            pids.append(i)

        faces = np.stack(faces)
        vert = np.stack(vert)
        
        for nv in self.view_list:
            basename = os.path.basename(img[nv])
            # 重新读入图片
            assert os.path.exists(img[nv]), img[nv]
            vis = cv2.imread(img[nv])
            vis = cv2.resize(vis, None, fx=self.scale, fy=self.scale)

            # vert = vertices
            meshes = {}
            if vert.ndim == 2:
                meshes[0] = {
                    'vertices': vert,
                    'faces': faces,
                    'id': 0,
                    'name': 'human_{}'.format(0)
                }
            elif vert.ndim == 3:
                for pid in range(vert.shape[0]):
                    meshes[pid] = {
                        'vertices': vert[pid],
                        'faces': faces[pid],
                        'vid': pids[pid],
                        'name': 'human_{}'.format(pid)
                    }
            K = cameras['K'][nv].copy()
            K[:2, :] *= self.scale
            R = cameras['R'][nv]
            T = cameras['T'][nv]
            # breakpoint()
            from easymocap.mytools.vis_base import plot_keypoints_auto
            for pid in range(keypoints3d.shape[0]):
                keypoints_repro = projectPoints(keypoints3d[pid].T[:3,:], K, R, T, cameras['dist'][nv].reshape(5)).T
                keypoints_repro[:,-1] = keypoints3d[pid,:,-1]
                plot_keypoints_auto(vis, keypoints_repro, pid=pid, use_limb_color=False)

            ret = plot_meshes(vis, meshes, K, R, T, mode=self.mode)
            outname = join(self.output, self.name, basename)
            os.makedirs(os.path.dirname(outname), exist_ok=True)
            cv2.imwrite(outname, ret)

class Render_selectview:
    def __init__(self, scale=0.5, backend='pyrender', output='output',mode = 'image') -> None:
        self.name = 'render_debug'
        self.scale = scale
        self.view_list = [5]
        self.output = output
        self.mode = mode

    def __call__(self, hand_model_l, posel, match3d_l, cameras, imgnames, keypoints3d,bbox_handl, joint_regressor, wristid):

        img = imgnames
        k3d = keypoints3d
        # joint_regressor_r = np.load('models/handmesh/data/joint_regressor_r.npy') #右手
        # joint_regressor_l = np.load('models/handmesh/data/joint_regressor_l.npy') 
        joint_regressor_l = joint_regressor
        facesl = hand_model_l.faces
        # facesr = hand_model_r.faces
        # breakpoint()
        hand_list=[]
        for pid in range(len(match3d_l)):
            dt = match3d_l[pid]
            if(isinstance(dt,int)):
                # TODO:处理-1的情况，也就是没有找到合适的匹配到的手
                hand_list.append(np.zeros((1,48)))
                break
            # Merge_list=[]
            out_img = []
            for cid in range(len(dt['views'])):
                nv = dt['views'][cid]
                poseid = dt['indices'][cid]
                pose = posel[nv][poseid].copy()

                Rh = pose[:,:3].copy()
                invR = np.linalg.inv(cameras['R'][nv])
                Rh_m_old = np.matrix(cv2.Rodrigues(Rh)[0])
                Rh_m_new = invR @ Rh_m_old
                Rh = cv2.Rodrigues(Rh_m_new)[0]
                
                pose_ = np.hstack((Rh.reshape(3),pose[:,3:].reshape(-1))).reshape(1,-1)

                Rh = pose_[:,:3].copy()
                pose_[:,:3] = 0
                params_l={
                    'Rh':Rh,
                    'Th':np.zeros_like(Rh),
                    'poses':pose_,
                    'shapes':np.zeros((Rh.shape[0],10)),
                }
                vertices_l = hand_model_l(**params_l, return_verts=True, return_tensor=False)
                joint_l = np.repeat(joint_regressor_l[None, :, :],vertices_l.shape[0],0) @ vertices_l
                params_l['Th']+=k3d[pid,wristid,:3] - joint_l[0,0,:]
                vertices_l = hand_model_l(**params_l, return_verts=True, return_tensor=False)

                vert = vertices_l[0]
                faces = facesl

                basename = os.path.basename(img[nv])
                # 重新读入图片
                assert os.path.exists(img[nv]), img[nv]
                vis = cv2.imread(img[nv])

                plot_bbox(vis,bbox_handl[nv][poseid],0)
                vis = cv2.resize(vis, None, fx=self.scale, fy=self.scale)

                meshes = {}
                if vert.ndim == 2:
                    meshes[0] = {
                        'vertices': vert,
                        'faces': faces,
                        'id': 0,
                        'name': 'human_{}'.format(0)
                    }
                elif vert.ndim == 3:
                    for pid in range(vert.shape[0]):
                        meshes[pid] = {
                            'vertices': vert[pid],
                            'faces': faces[pid],
                            'id': pid,
                            'name': 'human_{}'.format(pid)
                        }
                K = cameras['K'][nv].copy()
                K[:2, :] *= self.scale
                R = cameras['R'][nv]
                T = cameras['T'][nv]
                # breakpoint()
                ret = plot_meshes(vis, meshes, K, R, T, mode=self.mode)
                out_img.append(ret)
            
            out_img = merge(out_img)
            outname = join(self.output, self.name, '{}-{:02d}.jpg'.format(basename.split('.jpg')[0],pid))
            os.makedirs(os.path.dirname(outname), exist_ok=True)
            cv2.imwrite(outname, out_img)

class Render_selectview_lr:
    def __init__(self, scale=0.5, backend='pyrender', output='output',mode = 'image') -> None:
        self.output = output
        self.model_l = Render_selectview(scale=0.5, backend='pyrender', output = self.output,mode = mode)
        self.model_r = Render_selectview(scale=0.5, backend='pyrender', output = self.output,mode = mode)
        self.model_l.name+='_l'
        self.model_r.name+='_r'
    def __call__(self, hand_model_l, posel, poser, match3d_l, match3d_r, hand_model_r, cameras, imgnames, keypoints3d,bbox_handl,bbox_handr):
        joint_regressor_r = np.load('models/handmesh/data/joint_regressor_r.npy') #右手
        joint_regressor_l = np.load('models/handmesh/data/joint_regressor_l.npy') 

        self.model_l(hand_model_l, posel, match3d_l, cameras, imgnames, keypoints3d,bbox_handl, joint_regressor_l, 7)
        self.model_r(hand_model_r, poser, match3d_r, cameras, imgnames, keypoints3d,bbox_handr, joint_regressor_r, 4)

class Render_mv(Render):
    def __call__(self, body_model, params, cameras, imgnames):
        # breakpoint()
        super().__call__(body_model, params, cameras, [imgnames[0],imgnames[1]])