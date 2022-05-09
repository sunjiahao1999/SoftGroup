import os.path as osp
from glob import glob

import numpy as np
import torch

from ..ops import voxelization_idx
from .custom import CustomDataset
from .utils import Instances3D
import torch_scatter


class S3DISInstDataset(CustomDataset):

    CLASSES = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair', 'table',
               'bookcase', 'sofa', 'board', 'clutter')

    def get_filenames(self):
        if isinstance(self.prefix, str):
            self.prefix = [self.prefix]
        filenames_all = []
        for p in self.prefix:
            filenames = glob(osp.join(self.data_root, p + '*' + self.suffix))
            assert len(filenames) > 0, f'Empty {p}'
            filenames_all.extend(filenames)
        filenames_all = sorted(filenames_all * self.repeat)
        return filenames_all

    def load(self, filename):
        # TODO make file load results consistent
        xyz, rgb, semantic_label, instance_label, _, _ = torch.load(filename)
        superpoint_filename=filename.replace('preprocess','superpoint').replace(self.suffix, '.npy')
        superpoint = np.load(superpoint_filename)
        assert len(superpoint)==len(semantic_label)
        # subsample data
        if self.training:
            N = xyz.shape[0]
            inds = np.random.choice(N, int(N * 0.25), replace=False)
            xyz = xyz[inds]
            rgb = rgb[inds]
            semantic_label = semantic_label[inds]
            instance_label = self.getCroppedInstLabel(instance_label, inds)
            superpoint = np.unique(superpoint[inds],return_inverse=True)[1]
        return xyz, rgb, semantic_label, instance_label, superpoint

    def crop(self, xyz, step=64):
        return super().crop(xyz, step=step)

    def getInstance3D(self, instance_label, semantic_label, superpoint):
        num_insts = instance_label.max() + 1
        num_points = len(instance_label)
        gt_masks = torch.zeros(num_insts, num_points)
        gt_labels = torch.zeros(num_insts)
        for i in range(num_insts):
            idx = torch.where(instance_label == i)
            gt_masks[i][idx] = 1
            assert len(torch.unique(semantic_label[idx])) == 1
            gt_labels[i] = semantic_label[idx][0]
        insts = Instances3D(num_points)
        insts.gt_labels = gt_labels.long()
        gt_spmasks = torch_scatter.scatter_mean(gt_masks.float(), superpoint, dim=-1)
        insts.gt_spmasks = (gt_spmasks>0.5).float()
        # if not self.training:
        #     insts.gt_masks = gt_masks
        return insts
    
    def transform_train(self, xyz, rgb, semantic_label, instance_label, superpoint, aug_prob=0.9):
        xyz_middle = self.dataAugment(xyz, True, True, True, aug_prob)
        xyz = xyz_middle * self.voxel_cfg.scale
        if np.random.rand() < aug_prob:
            xyz = self.elastic(xyz, 6 * self.voxel_cfg.scale // 50, 40 * self.voxel_cfg.scale / 50)
            xyz = self.elastic(xyz, 20 * self.voxel_cfg.scale // 50,
                               160 * self.voxel_cfg.scale / 50)
        xyz_middle = xyz / self.voxel_cfg.scale
        xyz = xyz - xyz.min(0)
        max_tries = 5
        while (max_tries > 0):
            xyz_offset, valid_idxs = self.crop(xyz)
            if valid_idxs.sum() >= self.voxel_cfg.min_npoint:
                xyz = xyz_offset
                break
            max_tries -= 1
        if valid_idxs.sum() < self.voxel_cfg.min_npoint:
            return None
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        superpoint = np.unique(superpoint[valid_idxs],return_inverse=True)[1]
        return xyz, xyz_middle, rgb, semantic_label, instance_label, superpoint

    def transform_test(self, xyz, rgb, semantic_label, instance_label, superpoint):
        # devide into 4 piecies
        inds = np.arange(xyz.shape[0])
        piece_1 = inds[::4]
        piece_2 = inds[1::4]
        piece_3 = inds[2::4]
        piece_4 = inds[3::4]
        xyz_aug = self.dataAugment(xyz, False, False, False)

        xyz_list = []
        xyz_middle_list = []
        rgb_list = []
        semantic_label_list = []
        instance_label_list = []
        superpoint_list = []
        for batch, piece in enumerate([piece_1, piece_2, piece_3, piece_4]):
            xyz_middle = xyz_aug[piece]
            xyz = xyz_middle * self.voxel_cfg.scale
            xyz -= xyz.min(0)
            xyz_list.append(np.concatenate([np.full((xyz.shape[0], 1), batch), xyz], 1))
            xyz_middle_list.append(xyz_middle)
            rgb_list.append(rgb[piece])
            semantic_label_list.append(semantic_label[piece])
            instance_label_list.append(instance_label[piece])
            superpoint_list.append(superpoint[piece])
        xyz = np.concatenate(xyz_list, 0)
        xyz_middle = np.concatenate(xyz_middle_list, 0)
        rgb = np.concatenate(rgb_list, 0)
        semantic_label = np.concatenate(semantic_label_list, 0)
        instance_label = np.concatenate(instance_label_list, 0)
        superpoint = np.concatenate(superpoint_list, 0)
        superpoint = superpoint.astype(np.int64)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)  # TODO remove this
        return xyz, xyz_middle, rgb, semantic_label, instance_label, superpoint

    def __getitem__(self, index):
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, '')
        data = self.load(filename)
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        if data is None:
            return None
        xyz, xyz_middle, rgb, semantic_label, instance_label, superpoint = data
        info = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), semantic_label)
        inst_num, inst_pointnum, inst_cls, pt_offset_label = info
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb).float()
        if self.training:
            feat += torch.randn(3) * 0.1
        semantic_label = torch.from_numpy(semantic_label)
        instance_label = torch.from_numpy(instance_label)
        pt_offset_label = torch.from_numpy(pt_offset_label)
        superpoint = torch.from_numpy(superpoint)
        insts = self.getInstance3D(instance_label, semantic_label, superpoint)
        return (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num,
                inst_pointnum, inst_cls, pt_offset_label, superpoint, insts)

    def collate_fn(self, batch):
        if self.training:
            scan_ids = []
            coords = []
            coords_float = []
            feats = []
            semantic_labels = []
            instance_labels = []

            instance_pointnum = []  # (total_nInst), int
            instance_cls = []  # (total_nInst), long
            pt_offset_labels = []

            total_inst_num = 0
            batch_id = 0

            superpoints = []
            superpoint_bias = 0
            insts_list = []
            for data in batch:
                if data is None:
                    continue
                (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num,
                inst_pointnum, inst_cls, pt_offset_label, superpoint, insts) = data

                superpoint += superpoint_bias
                superpoint_bias = superpoint.max() + 1

                instance_label[np.where(instance_label != -100)] += total_inst_num
                total_inst_num += inst_num
                scan_ids.append(scan_id)
                coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
                coords_float.append(coord_float)
                feats.append(feat)
                semantic_labels.append(semantic_label)
                instance_labels.append(instance_label)
                instance_pointnum.extend(inst_pointnum)
                instance_cls.extend(inst_cls)
                pt_offset_labels.append(pt_offset_label)

                superpoints.append(superpoint)
                insts_list.append(insts)

                batch_id += 1
            assert batch_id > 0, 'empty batch'
            if batch_id < len(batch):
                self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

            # merge all the scenes in the batch
            coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
            batch_idxs = coords[:, 0].int()
            coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
            feats = torch.cat(feats, 0)  # float (N, C)
            semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)
            instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
            instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
            instance_cls = torch.tensor(instance_cls, dtype=torch.long)  # long (total_nInst)
            pt_offset_labels = torch.cat(pt_offset_labels).float()
            superpoints = torch.cat(superpoints, 0).long()  # long (N)

            spatial_shape = np.clip(
                coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
            voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
            return {
                'scan_ids': scan_ids,
                'coords': coords,
                'batch_idxs': batch_idxs,
                'voxel_coords': voxel_coords,
                'p2v_map': p2v_map,
                'v2p_map': v2p_map,
                'coords_float': coords_float,
                'feats': feats,
                'semantic_labels': semantic_labels,
                'instance_labels': instance_labels,
                'instance_pointnum': instance_pointnum,
                'instance_cls': instance_cls,
                'pt_offset_labels': pt_offset_labels,
                'spatial_shape': spatial_shape,
                'batch_size': batch_id,
                'superpoints': superpoints,
                'insts': insts_list,
            }

        # assume 1 scan only
        (scan_id, coord, coord_float, feat, semantic_label, instance_label, inst_num, inst_pointnum,
         inst_cls, pt_offset_label, superpoint, insts) = batch[0]
        scan_ids = [scan_id]
        coords = coord.long()
        batch_idxs = torch.zeros_like(coord[:, 0].int())
        coords_float = coord_float.float()
        feats = feat.float()
        semantic_labels = semantic_label.long()
        instance_labels = instance_label.long()
        instance_pointnum = torch.tensor([inst_pointnum], dtype=torch.int)
        instance_cls = torch.tensor([inst_cls], dtype=torch.long)
        pt_offset_labels = pt_offset_label.float()
        
        superpoints = superpoint.long()
        insts = [insts]

        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, 4)
        return {
            'scan_ids': scan_ids,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'instance_pointnum': instance_pointnum,
            'instance_cls': instance_cls,
            'pt_offset_labels': pt_offset_labels,
            'spatial_shape': spatial_shape,
            'batch_size': 4,
            'superpoints': superpoints,
            'insts': insts,
        }
