import pickle

import cv2
import numpy as np

from ...utils import box_utils
from .database_sampler import DataBaseSampler
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
import torch


class Cart2Spher(object):
    def __init__(self, proj_cfg):
        super(Cart2Spher, self).__init__()
        self.size_h, self.size_w = np.array(proj_cfg['size'], np.float32)
        self.fov_up = proj_cfg['fov_up'] / 180.0 * np.pi
        self.fov_down = proj_cfg['fov_down'] / 180.0 * np.pi
        self.fov = abs(self.fov_up) + abs(self.fov_down)

    def points3d_to_points2d(self, points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dist = np.linalg.norm(points, 2, axis=1)
        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / dist)
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov
        # scale to image size using angular resolution
        proj_x *= self.size_w  # in [0.0, W]
        proj_y *= self.size_h  # in [0.0, H]

        points = np.stack([proj_x, proj_y], axis=-1)
        return points

    def box3d_to_boxes2d(self, boxes3d):
        corners3d = box_utils.boxes_to_corners_3d(boxes3d)   # (N, 8, 3)
        corners2d = self.points3d_to_points2d(corners3d.reshape(-1, 3))     # (N*8, 2)
        corners2d = corners2d.reshape(-1, 8, 2)     # (N, 8, 2)
        min_uv = np.min(corners2d, axis=1)  # (N, 2)
        max_uv = np.max(corners2d, axis=1)  # (N, 2)
        boxes2d = np.concatenate([min_uv, max_uv], axis=1)    # (N, 4)

        return boxes2d

    def points_in_view(self, points, view):
        """
        Args:
            points: (N, 2)  (x, y)
            view:   (x1, y1, x2, y2)
        Returns:
        """
        x1, y1, x2, y2 = view
        points_2d = self.points3d_to_points2d(points)   # (N, 2)
        flag = (points_2d[:, 0] >= x1) & (points_2d[:, 0] < x2) & (points_2d[:, 1] >= y1) & (points_2d[:, 1] < y2)
        return flag


class CMDataBaseSampler(DataBaseSampler):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        super(CMDataBaseSampler, self).__init__(root_path=root_path, sampler_cfg=sampler_cfg,
                                                class_names=class_names, logger=logger)
        self.depth_consistent = self.sampler_cfg.DEPTH_CONSISTENT
        self.check_2D_collision = self.sampler_cfg.CHECK_2D_COLLISION
        self.collision_thr = self.sampler_cfg.COLLISION_THR
        self.blending_type = self.sampler_cfg.BLENDING_TYPE
        self.spher_proj = Cart2Spher(sampler_cfg['PROJECT_CFG'])

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    def paste_obj(self, img, obj_img, obj_mask, bbox_2d):
        """
        Args:
            img: (H, W, 3)
            obj_img:   (h, w, 3)
            obj_mask:  (h, w)
            bbox_2d:   (4, )
        Returns:
            img: (H, W, 3)
        """
        # paste the image patch back
        x1, y1, x2, y2 = bbox_2d
        # the bbox might exceed the img size because the img is different
        img_h, img_w = img.shape[:2]
        # w = np.maximum(min(x2, img_w - 1) - x1 + 1, 1)
        # h = np.maximum(min(y2, img_h - 1) - y1 + 1, 1)
        w = np.maximum(min(x2, img_w - 1) - x1, 1)
        h = np.maximum(min(y2, img_h - 1) - y1, 1)
        obj_img = obj_img[:h, :w]

        if obj_mask is None:
            img[y1:y1+h, x1:x1+w] = obj_img
        else:
            obj_mask = obj_mask[:h, :w]

            # choose a blend option
            if not self.blending_type:
                blending_op = 'none'
            else:
                blending_choice = np.random.randint(len(self.blending_type))
                blending_op = self.blending_type[blending_choice]

            if blending_op.find('poisson') != -1:
                # options: cv2.NORMAL_CLONE=1, or cv2.MONOCHROME_TRANSFER=3
                # cv2.MIXED_CLONE mixed the texture, thus is not used.
                if blending_op == 'poisson':
                    mode = np.random.choice([1, 3], 1)[0]
                elif blending_op == 'poisson_normal':
                    mode = cv2.NORMAL_CLONE
                elif blending_op == 'poisson_transfer':
                    mode = cv2.MONOCHROME_TRANSFER
                else:
                    raise NotImplementedError
                center = (int(x1 + w / 2), int(y1 + h / 2))
                img = cv2.seamlessClone(obj_img, img, obj_mask * 255, center, mode)
            else:
                if blending_op == 'gaussian':
                    obj_mask = cv2.GaussianBlur(
                        obj_mask.astype(np.float32), (5, 5), 2)
                elif blending_op == 'box':
                    obj_mask = cv2.blur(obj_mask.astype(np.float32), (3, 3))
                paste_mask = 1 - obj_mask
                img[y1:y1 + h,
                    x1:x1 + w] = (img[y1:y1 + h, x1:x1 + w].astype(np.float32) *
                                  paste_mask[..., None]).astype(np.uint8)
                img[y1:y1 + h, x1:x1 + w] += (obj_img.astype(np.float32) *
                                              obj_mask[..., None]).astype(np.uint8)

        return img

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes_3d, sampled_gt_boxes_2d,
                                   sampled_boxes_spher, total_valid_sampled_dict):
        """
        Args:
            data_dict:
            sampled_gt_boxes_3d: (N_sample, 7)
            sampled_gt_boxes_2d: (N_sample, 4)
            sampled_boxes_spher: (N_sample, 4)
            total_valid_sampled_dict: List(dbinfo1, dbinfo2, ...)  len=N_sample
        Returns:

        """
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_boxes2d = data_dict['gt_boxes2d'][gt_boxes_mask]
        gt_boxes2d_copy = gt_boxes2d.copy()
        gt_boxes_spher = data_dict['gt_boxes_spher'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        img = data_dict['images']
        num_origin = gt_boxes.shape[0]
        # from tools.visualizer import open3d_vis
        # open3d_vis.show_result(points=points, gt_bboxes=gt_boxes, show=True)

        obj_points_list = []
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            obj_points[:, :3] += info['box3d_lidar'][:3]
            obj_points_list.append(obj_points)
        obj_points = np.concatenate(obj_points_list, axis=0)

        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes_3d[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)     # (N_points, 3)

        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)     # (N+N_sample, )
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes_3d], axis=0)      # (N+N_sample, 7)
        gt_boxes2d = np.concatenate([gt_boxes2d, sampled_gt_boxes_2d], axis=0)      # (N+N_sample, 4)
        gt_boxes_spher = np.concatenate([gt_boxes_spher, sampled_boxes_spher], axis=0)      # (N+N_sample, 4)

        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_cpu(
            points[..., :3], gt_boxes[:, 0:7]
        )  # (nboxes, npoints)

        fg_flag = box_idxs_of_pts.any(axis=0)
        bg_flag = ~fg_flag
        fg_points = points[fg_flag]     # (N_fg, 3)
        bg_points = points[bg_flag]     # (N_bg, 3)

        if self.depth_consistent:
            center = gt_boxes[:, 0:3]    # (N+N_sample, 3)
            distance = np.power(np.sum(np.power(center, 2), axis=-1), 1/2)  # (N+N_sample, )
            near_to_far_order = np.argsort(distance, axis=-1)
            far_to_near_order = near_to_far_order[::-1]

        pasted_points = obj_points
        for idx in range(len(gt_boxes)):
            origin_flag = False
            if self.depth_consistent:
                inds = near_to_far_order[idx]
            else:
                inds = idx

            if inds < num_origin:
                origin_flag = True
            cur_gt_boxes_spher = gt_boxes_spher[inds]
            cur_distance = distance[inds]

            if origin_flag:
                # If an original object is taken, we only discard those occluded points that belong to
                # farther pasted objects.

                view_flag = self.spher_proj.points_in_view(pasted_points, cur_gt_boxes_spher)   # (N_sample, )
                points_dist = np.power(np.sum(np.power(pasted_points, 2), axis=-1), 1/2)  # (N_sample, )
                dist_flag = pasted_points_dist >= cur_distance  # (N_sample, )
                flag = ~(view_flag & dist_flag)
                pasted_points = pasted_points[flag]
            else:
                # If a pasted object is processed, all occluded points farther than this object will be disposed.
                # Moreover, we filter the background points in the perspective of this virtual object.
                # 1. remove pasted_points
                view_flag = self.spher_proj.points_in_view(pasted_points, cur_gt_boxes_spher)  # (N_sample, )
                pasted_points_dist = np.power(np.sum(np.power(pasted_points, 2), axis=-1), 1 / 2)  # (N_sample, )
                dist_flag = pasted_points_dist >= cur_distance  # (N_sample, )
                flag = ~(view_flag & dist_flag)
                pasted_points = pasted_points[flag]
                # 2. remove fg_points
                view_flag = self.spher_proj.points_in_view(fg_points, cur_gt_boxes_spher)  # (N_sample, )
                fg_points_dist = np.power(np.sum(np.power(fg_points, 2), axis=-1), 1 / 2)  # (N_sample, )
                dist_flag = fg_points_dist >= cur_distance  # (N_sample, )
                flag = ~(view_flag & dist_flag)
                fg_points = fg_points[flag]
                # 3. remove bg_points
                view_flag = self.spher_proj.points_in_view(bg_points, cur_gt_boxes_spher)  # (N_sample, )
                flag = ~view_flag
                bg_points = bg_points[flag]

        points = np.concatenate([pasted_points, fg_points, bg_points], axis=0)
        # from tools.visualizer import open3d_vis
        # open3d_vis.show_result(points=points, gt_bboxes=gt_boxes, show=True)

        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_cpu(
            points[..., :3], gt_boxes[:, 0:7]
        )  # (nboxes, npoints)
        box_have_points_flag = box_idxs_of_pts.any(axis=1)  # (nboxes, )
        gt_boxes = gt_boxes[box_have_points_flag]
        gt_boxes2d = gt_boxes2d[box_have_points_flag]
        gt_names = gt_names[box_have_points_flag]
        original_idxs = box_have_points_flag.nonzero()[0]
        # from tools.visualizer import open3d_vis
        # open3d_vis.show_result(points=points, gt_bboxes=gt_boxes, show=True)

        img_copy = img.copy()
        for idx in range(len(gt_boxes2d)):
            origin_flag = False
            original_idx = original_idxs[idx]
            if self.depth_consistent:
                ind = far_to_near_order[original_idx]
            else:
                ind = original_idx

            if ind < num_origin:
                origin_flag = True

            if origin_flag:
                bbox = gt_boxes2d_copy[ind]    # (x1, y1, x2, y2)
                bbox = bbox.astype(np.int32)
                x1, y1, x2, y2 = bbox
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)
                img_patch = img_copy[y1:y1 + h, x1:x1 + w]
                s_mask = None
            else:
                info = total_valid_sampled_dict[ind - num_origin]
                bbox = info['bbox'].astype(np.int32)
                pcd_file_path = self.root_path / info['path']
                img_file_path = str(pcd_file_path) + '.png'
                mask_file_path = str(pcd_file_path) + '.mask.png'
                img_patch = cv2.imread(img_file_path)
                s_mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)

            img = self.paste_obj(
                img=img,
                obj_img=img_patch,
                obj_mask=s_mask,
                bbox_2d=bbox,
            )

        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_boxes2d'] = gt_boxes2d
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points

        calib = data_dict['calib']
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        pts_img, _ = calib.rect_to_img(pts_rect)
        data_dict['points_img_origin'] = pts_img

        # cv2.imshow("a", img)
        # cv2.waitKey(0)
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_boxes_2d: (N, 4)
        Returns:

        """
        gt_boxes_3d = data_dict['gt_boxes']     # (N, 7)
        gt_boxes_spher = self.spher_proj.box3d_to_boxes2d(gt_boxes_3d)  # (N, 4)    [x1, y1, x2, y2]
        data_dict['gt_boxes_spher'] = gt_boxes_spher
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes_3d = gt_boxes_3d      # (N, 7)
        existed_boxes_spher = gt_boxes_spher    # (N, 4)
        total_valid_sampled_dict = []
        total_valid_sampled_boxes3d = []
        total_valid_sampled_boxes2d = []
        total_valid_sampled_boxes_spher = []
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes_3d = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)   # (N', 7)
                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes_3d = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes_3d)
                sampled_boxes_2d = np.stack([x['bbox'] for x in sampled_dict], axis=0)  # (N', 4)

                existed_boxes_bv = box_utils.boxes_to_corners_2d(existed_boxes_3d)     # (N, 4, 2)
                sampled_boxes_bv = box_utils.boxes_to_corners_2d(sampled_boxes_3d)     # (N', 4, 2)
                total_boxes_bv = np.concatenate([existed_boxes_bv, sampled_boxes_bv], axis=0)     # (M=N+N', 4, 2)
                coll_mat = box_utils.box_collision_test(total_boxes_bv, total_boxes_bv)     # (M, M)

                # Then avoid collision in 2D spherical view
                if self.check_2D_collision:
                    sampled_boxes_spher = self.spher_proj.box3d_to_boxes2d(sampled_boxes_3d)    # (N', 4)
                    total_boxes_spher = np.concatenate([existed_boxes_spher, sampled_boxes_spher], axis=0)   # (M=N+N', 4)  (x1, y1, x2, y2)

                    if isinstance(self.collision_thr, float):
                        collision_thr = self.collision_thr
                    elif isinstance(self.collision_thr, list):
                        collision_thr = np.random.choice(self.collision_thr)
                    elif isinstance(self.collision_thr, dict):
                        mode = self.collision_thr.get('mode', 'value')
                        if mode == 'value':
                            collision_thr = np.random.choice(
                                self.collision_thr['thr_range'])
                        elif mode == 'range':
                            collision_thr = np.random.uniform(
                                self.collision_thr['thr_range'][0],
                                self.collision_thr['thr_range'][1])

                    if collision_thr == 0:
                        # use similar collision test as BEV did
                        # Nx4 (x1, y1, x2, y2) -> corners: Nx4x2
                        # ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
                        x1y1 = total_boxes_spher[:, :2]
                        x2y2 = total_boxes_spher[:, 2:]
                        x1y2 = np.stack([total_boxes_spher[:, 0], total_boxes_spher[:, 3]],
                                        axis=-1)
                        x2y1 = np.stack([total_boxes_spher[:, 2], total_boxes_spher[:, 1]],
                                        axis=-1)
                        total_2d = np.stack([x1y1, x2y1, x1y2, x2y2], axis=1)   # (M, 4, 2)
                        coll_mat_2d = box_utils.box_collision_test(
                            total_2d, total_2d)     # (M, M)
                    else:
                        # use iof rather than iou to protect the foreground
                        overlaps = box_utils.iou_jit(total_boxes_spher, total_boxes_spher, 'iof')
                        coll_mat_2d = overlaps > collision_thr      # (M, M)
                    coll_mat = coll_mat + coll_mat_2d   # (M, M)

                    # checks whether the img patch exceed the image boundary
                    image = data_dict['images']     # (H, W, 3)
                    h, w = image.shape[:2]
                    exceed_flag = (sampled_boxes_2d[:, 0] >= w-1) + (sampled_boxes_2d[:, 1] >= h-1)    # (N', )
                    coll_mat[-sampled_boxes_2d.shape[0]:, :] += exceed_flag[:, np.newaxis]  # (N', M)

                diag = np.arange(total_boxes_bv.shape[0])   # (M, )
                coll_mat[diag, diag] = False

                valid_sampled_dict = []
                valid_sampled_boxes_3d = []
                valid_sampled_boxes_2d = []
                valid_sampled_boxes_spher = []
                cur_existed_num = existed_boxes_3d.shape[0]
                cur_sampled_num = sampled_boxes_3d.shape[0]
                for i in range(cur_existed_num, cur_existed_num + cur_sampled_num):
                    if coll_mat[i].any():
                        coll_mat[i] = False
                        coll_mat[:, i] = False
                    else:
                        valid_sampled_dict.append(sampled_dict[i - cur_existed_num])
                        valid_sampled_boxes_3d.append(sampled_boxes_3d[i - cur_existed_num])    # (7, )
                        valid_sampled_boxes_2d.append(sampled_boxes_2d[i - cur_existed_num])    # (4, )
                        valid_sampled_boxes_spher.append(sampled_boxes_spher[i - cur_existed_num])      # (4, )

                if len(valid_sampled_dict) > 0:
                    valid_sampled_boxes_3d = np.stack(valid_sampled_boxes_3d, axis=0)   # (N_sample, 7)
                    valid_sampled_boxes_2d = np.stack(valid_sampled_boxes_2d, axis=0)   # (N_sample, 4)
                    valid_sampled_boxes_spher = np.stack(valid_sampled_boxes_spher, axis=0)     # (N_sample, 4)
                    total_valid_sampled_boxes3d.append(valid_sampled_boxes_3d)
                    total_valid_sampled_boxes2d.append(valid_sampled_boxes_2d)
                    total_valid_sampled_boxes_spher.append(valid_sampled_boxes_spher)
                    total_valid_sampled_dict.extend(valid_sampled_dict)

                    existed_boxes_3d = np.concatenate((existed_boxes_3d, valid_sampled_boxes_3d), axis=0)
                    existed_boxes_spher = np.concatenate((existed_boxes_spher, valid_sampled_boxes_spher), axis=0)

                # print(class_name, len(valid_sampled_dict))

        if total_valid_sampled_dict.__len__() > 0:
            total_valid_sampled_boxes3d = np.concatenate(total_valid_sampled_boxes3d, axis=0)
            total_valid_sampled_boxes2d = np.concatenate(total_valid_sampled_boxes2d, axis=0)
            total_valid_sampled_boxes_spher = np.concatenate(total_valid_sampled_boxes_spher, axis=0)
            data_dict = self.add_sampled_boxes_to_scene(data_dict, total_valid_sampled_boxes3d, total_valid_sampled_boxes2d,
                                                        total_valid_sampled_boxes_spher, total_valid_sampled_dict)

        data_dict.pop('gt_boxes_mask')
        data_dict.pop('gt_boxes_spher')
        return data_dict
