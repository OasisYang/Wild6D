import os
import cv2
import math
import random
import numpy as np
import json
import gzip
import _pickle as cPickle
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from lib.utils import load_depth, get_bbox, load_obj, farthest_point_sampling, load_depth_co3d, _load_1bit_png_mask
from lib.transformations import quaternion_from_matrix
import pdb

class PoseDataset(data.Dataset):
    def __init__(self, source, mode, data_dir, n_pts, img_size, select_class='all', 
        use_dz=False, use_co3d=False, use_ucsd_pose=False, semi_percent=1.0, no_depth=False):
        """
        Args:
            source: 'CAMERA', 'Real' or 'CAMERA+Real'
            mode: 'train' or 'test'
            data_dir:
            n_pts: number of selected foreground points
            img_size: square image window
        """
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size
        self.select_class = select_class
        self.use_dz = use_dz
        self.use_co3d = use_co3d
        self.use_ucsd_pose = use_ucsd_pose
        self.no_depth = no_depth 

        assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test']
        img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'Real/test_list.txt', 'Real/test_list.txt']
        model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
                           'obj_models/real_test.pkl', 'obj_models/real_test.pkl']
        if mode == 'train':
            del img_list_path[2:]
            del model_file_path[2:]
        else:
            del img_list_path[:2]
            del model_file_path[:2]
        if source == 'CAMERA':
            del img_list_path[-1]
            del model_file_path[-1]
        elif source == 'Real':
            del img_list_path[0]
            del model_file_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del img_list_path[0]
                del model_file_path[0]

        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']


        img_list = []
        subset_len = []
        for path in img_list_path:
            if self.select_class == 'all':
                img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                            for line in open(os.path.join(data_dir, path))]
            else:
                self.select_class_id = self.cat_names.index(self.select_class) + 1
                img_cate_dict = cPickle.load(open('assets/img_cate_label.pkl', 'rb'))
                img_n = [os.path.join(path.split('/')[0], line.rstrip('\n'))
                            for line in open(os.path.join(data_dir, path))]
                img_list += self._filter_by_class(img_n, img_cate_dict)  
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1]-subset_len[0]]
        self.img_list = img_list
        if self.use_co3d:
            self.__init_co3d__()
        if self.use_ucsd_pose:
            self.semi_percent = semi_percent
            self.__init_ucsd_pose__(self.select_class, self.semi_percent)

        self.length = len(self.img_list)

        models = {}
        for path in model_file_path:
            with open(os.path.join(data_dir, path), 'rb') as f:
                models.update(cPickle.load(f))
        self.models = models

        # meta info for re-label mug category
        with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        self.mean_shapes = np.load('data/meshes/mean_points_emb.npy')
        # mean_shape_path = 'assets/mean_shape_mesh.pkl'
        # self.mean_meshes = cPickle.load(open(mean_shape_path, 'rb'))
        

        self.camera_intrinsics = [577.5, 577.5, 319.5, 239.5]    # [fx, fy, cx, cy]
        self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.shift_range = 0.01
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        print('{} images found.'.format(self.length))
        print('{} models loaded.'.format(len(self.models)))


    def __init_ucsd_pose__(self, select_class, semi_percent):
        self.ucsd_pose_dir = os.path.join('data/Wild6D', select_class)
        img_list = [os.path.join(self.ucsd_pose_dir, line.strip('\n')) for line in \
            open(os.path.join(self.ucsd_pose_dir, 'train_list.txt'))]
        # self.ucsd_pose_dir = os.path.join('data/UCSD_POSE_RGBD/test_set', select_class)
        # img_list = [line.strip('\n') for line in open(os.path.join(self.ucsd_pose_dir, 'train_list.txt'))]
        num_to_use = int(len(img_list) * semi_percent)
        print("Use {} real RGBD images".format(num_to_use))
        self.ucsd_img_list = img_list[:num_to_use]


    def __len__(self):
        return self.length

    def _filter_by_class(self, img_list, img_cate_dict):
        img_list_new = []
        for img in img_list:
            objs = img_cate_dict[img]
            if self.select_class_id in objs:
                img_list_new.append(img)
        return img_list_new


    def xywh_to_cs_dzi(self, bbox, s_ratio, s_max=None, tp='uniform'):
        x, y, w, h = bbox
        if tp == 'gaussian':
            sigma = 1
            shift = truncnorm.rvs(-self.cfg.augment.shift_ratio / sigma, self.cfg.augment.shift_ratio / sigma, scale=sigma, size=2)
            scale = 1+truncnorm.rvs(-self.cfg.augment.scale_ratio / sigma, self.cfg.augment.scale_ratio / sigma, scale=sigma, size=1)
        elif tp == 'uniform':
            scale = 1+0.25 * (2*np.random.random_sample()-1)
            shift = 0.25 * (2*np.random.random_sample(2)-1)
        c = np.array([x+w*(0.5+shift[1]), y+h*(0.5+shift[0])]) # [c_w, c_h]
        s = max(w, h)*s_ratio*scale
        if s_max != None:
            s = float(min(s, s_max))
        return c, s

    def xywh_to_cs(self, bbox, s_ratio, s_max=None):
        x, y, w, h = bbox
        c = np.array((x+0.5*w, y+0.5*h)) # [c_w, c_h]
        s = max(w, h)*s_ratio
        if s_max != None:
            s = float(min(s, s_max))
        return c, s
    
    def zoom_in(self, im, c, s, res, channel=3, interpolate=cv2.INTER_LINEAR):
        """
        zoom in on the object with center c and size s, and resize to resolution res.
        :param im: nd.array, single-channel or 3-channel image
        :param c: (w, h), object center
        :param s: scalar, object size
        :param res: target resolution
        :param channel:
        :param interpolate:
        :return: zoomed object patch 
        """
        c_w, c_h = c
        c_w, c_h, s, res = int(c_w), int(c_h), int(s), int(res)
        if channel==1:
            im = im[..., None]
        h, w = im.shape[:2]
        u = int(c_h-0.5*s+0.5)
        l = int(c_w-0.5*s+0.5)
        b = u+s
        r = l+s
        if (u>=h) or (l>=w) or (b<=0) or (r<=0):
            return np.zeros((res, res, channel)).squeeze()
        if u < 0:
            local_u = -u
            u = 0 
        else:
            local_u = 0
        if l < 0:
            local_l = -l
            l = 0
        else:
            local_l = 0
        if b > h:
            local_b = s-(b-h)
            b = h
        else:
            local_b = s
        if r > w:
            local_r = s-(r-w)
            r = w
        else:
            local_r = s
        # im_crop = np.zeros((s, s, channel))
        # im_crop[local_u:local_b, local_l:local_r, :] = im[u:b, l:r, :]
        im_crop = im[u:b, l:r, :]
        # im_crop = im_crop.squeeze()
        im_resize = cv2.resize(im_crop, (res, res), interpolation=interpolate)
        c_h = 0.5*(u+b)
        c_w = 0.5*(l+r)
        s = s
        return im_resize, c_h, c_w, s, np.array([u, b, l, r])

    def c_rel_delta(self, c_obj, c_box, wh_box):
        """
        compute relative bias between object center and bounding box center
        """
        c_delta = np.asarray(c_obj) - np.asarray(c_box)
        c_delta /= np.asarray(wh_box)
        return c_delta

    def d_scaled(self, depth, s_box, res):
        """
        compute scaled depth
        """
        r = float(res) / s_box
        return depth / r

    def proj_cam(self, vtx, cam_K):
        sp = vtx.shape
        if (len(sp) == 1) and (sp[0] == 3):
            single = True
        elif (len(sp) == 2) and (sp[1] == 3):
            single = False
        else:
            raise
        if single:
            vtx = np.asarray(vtx)[None, :] 
        pts_3d_c = np.matmul(cam_K, vtx.T) 
        pts_2d = pts_3d_c[:2] / pts_3d_c[2]
        z = pts_3d_c[2]
        if single:
            return pts_2d.squeeze(), z
        else:
            return pts_2d.T, z

    def compute_T(self, trans, s_box, c_box, bbox, K):
        # compute T from translation head
        ratio_delta_c = trans[:2]
        ratio_depth = trans[2]
        pred_depth = ratio_depth * (self.opts.img_size / s_box)
        pred_c = ratio_delta_c * bbox[2:] + c_box
        pred_x = (pred_c[0] - K[0, 2]) * pred_depth / K[0, 0]
        pred_y = (pred_c[1] - K[1, 2]) * pred_depth / K[1, 1]
        return np.asarray([pred_x, pred_y, pred_depth])

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_list[index])
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1]
        depth = load_depth(img_path)
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2]
        coord = cv2.imread(img_path + '_coord.png')[:, :, :3]
        coord = coord[:, :, (2, 1, 0)]
        coord = np.array(coord, dtype=np.float32) / 255
        coord[:, :, 2] = 1 - coord[:, :, 2]
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        if 'CAMERA' in img_path.split('/'):
            cam_fx, cam_fy, cam_cx, cam_cy = self.camera_intrinsics
            is_real = False
        else:
            cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics
            is_real = True

        # select one foreground object
        if self.select_class == 'all':
            idx = random.randint(0, len(gts['instance_ids'])-1)
        else:
            idx_candidate = [idx for idx, obj_cate in enumerate(gts['class_ids']) 
                            if obj_cate == self.select_class_id]
            idx = random.choice(idx_candidate)

        inst_id = gts['instance_ids'][idx]
        y1, x1, y2, x2 = gts['bboxes'][idx]
        bbox = np.array([x1, y1, x2-x1, y2-y1]).astype(np.float32)
        
        # rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        # sample points
        mask = np.equal(mask, inst_id).astype(np.float32)
        # mask = np.logical_and(mask, depth > 0).astype(np.float32)
        # resize and crop obj
        if self.mode == 'train' and self.use_dz:
            c, s = self.xywh_to_cs_dzi(bbox, 1.5, s_max=max(640, 480))
        else:
            c, s = self.xywh_to_cs(bbox, 1.5, s_max=max(640, 480))
        rgb, c_h_, c_w_, s_, crop_bbox = self.zoom_in(rgb, c, s, self.img_size)
        rmin, rmax, cmin, cmax = crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3]
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        mask, *_ = self.zoom_in(mask, c, s, self.img_size, channel=1, interpolate=cv2.INTER_NEAREST)
        if len(choose) > self.n_pts:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.n_pts] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.n_pts-len(choose)), 'wrap')
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        pt2 = depth_masked / self.norm_scale
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        points = np.concatenate((pt0, pt1, pt2), axis=1)
        nocs = coord[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :] - 0.5
        # resize cropped image to standard size and adjust 'choose' accordingly
        # rgb = rgb[rmin:rmax, cmin:cmax, :]
        # rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        crop_w = cmax - cmin
        crop_h = rmax - rmin
        ratio_w = self.img_size / float(crop_w)
        ratio_h = self.img_size / float(crop_h)
        choose_max = np.max(choose)
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio_h) * self.img_size + np.floor(col_idx * ratio_w)).astype(np.int64)
        kpt_points = np.stack((np.floor(row_idx * ratio_h), np.floor(col_idx * ratio_w)), axis=1).astype(np.int64)
        kpt_choose = farthest_point_sampling(kpt_points, 32)
        kpt_points = kpt_points[kpt_choose]
        # kpt_choose = kpt_points[:, 0] * self.img_size + kpt_points[:, 1]
        kpt_points = kpt_points / self.img_size

        if np.max(choose) > np.square(self.img_size):
            print(choose_max)
            print(crop_bbox)
            print(ratio)
            print(index)
        
        # label
        cat_id = gts['class_ids'][idx] - 1    # convert to 0-indexed
        model = self.models[gts['model_list'][idx]].astype(np.float32)     # 1024 points
        prior = self.mean_shapes[cat_id].astype(np.float32)
        
        path_to_mesh_model = 'data/meshes/{}.obj'.format(self.cat_names[cat_id])
        verts = load_obj(path_to_mesh_model)[0]
        # verts, faces = mean_meshes

        # mean_shape = (verts, faces)
        scale = gts['scales'][idx]
        rotation = gts['rotations'][idx]
        translation = gts['translations'][idx]
        # data augmentation
        if self.mode == 'train':
            # color jitter
            rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
            rgb = np.array(rgb)
            # # point shift
            # add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
            # translation = translation + add_t[0]
            # # point jitter
            # add_t = add_t + np.clip(0.001*np.random.randn(points.shape[0], 3), -0.005, 0.005)
            # points = np.add(points, add_t)
        rgb = self.transform(rgb)
        points = points.astype(np.float32)
        # adjust nocs coords for mug category
        if cat_id == 5:
            T0 = self.mug_meta[gts['model_list'][idx]][0]
            s0 = self.mug_meta[gts['model_list'][idx]][1]
            nocs = s0 * (nocs + T0)
        # map ambiguous rotation to canonical rotation
        if cat_id in self.sym_ids:
            rotation = gts['rotations'][idx]
            # assume continuous axis rotation symmetry
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x**2 + theta_y**2)
            s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                              [0.0,            1.0,  0.0           ],
                              [theta_y/r_norm, 0.0,  theta_x/r_norm]])
            rotation = rotation @ s_map
            nocs = nocs @ s_map
        sRT = np.identity(4, dtype=np.float32)
        sRT[:3, :3] = scale * rotation
        sRT[:3, 3] = translation
        nocs = nocs.astype(np.float32)

        quat_matrix = quaternion_from_matrix(rotation)
        pose = np.zeros((8), dtype=np.float32)
        pose[0] = scale
        pose[1:4] = translation
        pose[4:] = quat_matrix

        cam = np.identity(3, dtype=np.float32)
        cam[0, 0] = cam_fx
        cam[1, 1] = cam_fy
        cam[0, 2] = cam_cx
        cam[1, 2] = cam_cy

        # compute delta for translation
        box_c = np.array([c_w_, c_h_])
        box_s = s_
        c_obj, _ = self.proj_cam(translation, cam)
        c_delta = self.c_rel_delta(c_obj, box_c, bbox[2:])
        d_local = self.d_scaled(translation[2], box_s, self.img_size)
        trans_local = np.append(c_delta, [d_local], axis=0).astype(np.float32)
        if self.cat_names[cat_id] in ['bottle', 'can', 'mug', 'bowl']:
            is_symmetric = True
        else:
            is_symmetric = False

        if self.select_class != 'all':
            cat_id = 0
        
        if self.use_ucsd_pose:
            ucsd_pose_data = self.load_ucsd_pose(index)
            return points, rgb, mask, choose, cat_id, model, prior, sRT, verts, \
                nocs, pose, trans_local, bbox, box_c, box_s, crop_bbox, cam, is_real, \
                is_symmetric, ucsd_pose_data
        else:
            return points, rgb, mask, choose, cat_id, model, prior, sRT, verts, \
                nocs, pose, trans_local, bbox, box_c, box_s, crop_bbox, cam, is_real, \
                is_symmetric

    def get_camera(self, principal_point, focal_length, img_size):
        h, w = img_size
        half_image_size_wh_orig = [w/2., h/2.]
        principal_point_px = -1.0 * (principal_point - 1.0) * half_image_size_wh_orig
        focal_length_px = focal_length * half_image_size_wh_orig
        cam = np.identity(3, dtype=np.float32)
        cam[0, 0] = focal_length_px[0]
        cam[1, 1] = focal_length_px[1]
        cam[0, 2] = principal_point_px[0]
        cam[1, 2] = principal_point_px[1]

        return cam

    def load_ucsd_pose(self, idx):
        flag = 1
        while flag:
            random.shuffle(self.ucsd_img_list)
            if idx >= len(self.ucsd_img_list):
                idx = idx % len(self.ucsd_img_list)
            # img_idx = np.random.choice(len(self.ucsd_img_list))
            img_path = self.ucsd_img_list[idx]
            mask_path = img_path.replace('.jpg', '-mask.png')
            depth_path = img_path.replace('.jpg', '-depth.png')
            assert os.path.exists(img_path)
            assert os.path.exists(depth_path)

            if not os.path.exists(mask_path):
                continue

            vid_name = os.path.join(img_path.split('/')[-4], img_path.split('/')[-3])
            meta_data = json.load(open(os.path.join(self.ucsd_pose_dir, vid_name, 'metadata')))
            cam = np.array(meta_data['K']).reshape(3, 3).T
            # cam_fy = K[0, 0]
            # cam_fx = K[1, 1]
            # cam_cy = K[0, 2] 
            # cam_cx = K[1, 2]
            # cam = np.identity(3, dtype=np.float32)

            w = meta_data['w']
            h = meta_data['h']

            rgb = cv2.imread(img_path)[:, :, :3]
            rgb = rgb[:, :, ::-1]
            try:
                mask = cv2.imread(mask_path)[:, :, 2] 
            except:
                print(mask_path)
            mask = mask / 255.
            depth = cv2.imread(depth_path, -1)
            mask_new = np.logical_and(mask, depth > 0).astype(np.float32)
            
            if mask.shape != depth.shape:
                print("mask shape:{}".format(mask.shape))
                print("depth shape:{}".format(depth.shape))
            inst_id = 1
            # bounding box
            horizontal_indicies = np.where(np.any(mask, axis=0))[0]
            vertical_indicies = np.where(np.any(mask, axis=1))[0]
            try:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
            except:
                print(mask_path)
                continue

            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            bbox = np.array([x1, y1, x2-x1, y2-y1]).astype(np.float32)
            c, s = self.xywh_to_cs(bbox, 1.2, s_max=max(w, h))
            rgb, c_h_, c_w_, s_, crop_bbox = self.zoom_in(rgb, c, s, self.img_size)
            # cv2.imwrite('test.jpg', rgb)
            rmin, rmax, cmin, cmax = crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3]
            choose = mask_new[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) == 0:
                print(mask_path)
                continue
            assert len(choose) > 0
            mask, *_ = self.zoom_in(mask, c, s, self.img_size, channel=1, interpolate=cv2.INTER_NEAREST)
            # cv2.imwrite('test-mask.jpg', mask)
            if len(choose) > self.n_pts:
                c_mask = np.zeros(len(choose), dtype=np.int32)
                c_mask[:self.n_pts] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.n_pts-len(choose)), 'wrap')

            xmap = np.array([[i for i in range(w)] for j in range(h)])
            ymap = np.array([[j for i in range(w)] for j in range(h)])
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            
            crop_w = cmax - cmin
            crop_h = rmax - rmin
            ratio_w = self.img_size / float(crop_w)
            ratio_h = self.img_size / float(crop_h)
            col_idx = choose % crop_w
            row_idx = choose // crop_w
            choose = (np.floor(row_idx * ratio_h) * self.img_size + np.floor(col_idx * ratio_w)).astype(np.int64)

            cam_fx = cam[0, 0]
            cam_fy = cam[1, 1]
            cam_cx = cam[0, 2] 
            cam_cy = cam[1, 2]
            pt2 = depth_masked / self.norm_scale
            pt0 = (xmap_masked - cam_cx) / cam_fx
            pt1 = (ymap_masked - cam_cy) / cam_fy
            points = np.concatenate((pt0, pt1, pt2), axis=1)

            box_c = np.array([c_w_, c_h_])
            box_s = s_
            if self.mode == 'train':
                # color jitter
                rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
                rgb = np.array(rgb)
            rgb = self.transform(rgb)
            points = points.astype(np.float32)
            is_real = True
            flag = 0
        return points, rgb, mask, choose, cam, bbox, crop_bbox, box_c, box_s, is_real