import glob
import os
import pickle
import numpy as np
import open3d as o3d
from nuscenes.utils.data_classes import LidarPointCloud
import yaml
from PIL import Image
import xml.etree.ElementTree as ET

from lidm.data.base import DatasetBase
from .annotated_dataset import Annotated3DObjectsDataset
from .conditional_builder.utils import corners_3d_to_2d
from .helper_types import Annotation
from ..utils.lidar_utils import pcd2range, pcd2coord2d, range2pcd

# TODO add annotation categories and semantic categories

# train + test

NUSCENES_TRAIN_SET = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

# validation


class NUSCENESBase(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = 'nuscenes'
        self.num_sem_cats = kwargs['dataset_config'].num_sem_cats + 1

    @staticmethod
    def load_lidar_sweep(path):
        pc = LidarPointCloud.from_file(path)
        points = pc.points.T
        points = points.reshape((-1, 4))[:, 0:3]
        #print(points.shape)
        return points

    def load_semantic_map(self, path, pcd):
        raise NotImplementedError

    def load_camera(self, path):
        raise NotImplementedError

    def __getitem__(self, idx):
        example = dict()
        data_path = self.data[idx]
        # lidar point cloud
        sweep = self.load_lidar_sweep(data_path)

        if self.lidar_transform:
            sweep, _ = self.lidar_transform(sweep, None)

        if self.condition_key == 'segmentation':
            # semantic maps
            proj_range, sem_map = self.load_semantic_map(data_path, sweep)
            example[self.condition_key] = sem_map
        else:
            proj_range, _ = pcd2range(sweep, self.img_size, self.fov, self.depth_range)
        proj_range, proj_mask = self.process_scan(proj_range)
        example['image'], example['mask'] = proj_range, proj_mask
        if self.return_pcd:
            reproj_sweep, _, _ = range2pcd(proj_range[0] * .5 + .5, self.fov, self.depth_range, self.depth_scale, self.log_scale)
            example['raw'] = sweep
            example['reproj'] = reproj_sweep.astype(np.float32)

        # image degradation
        if self.degradation_transform:
            degraded_proj_range = self.degradation_transform(proj_range)
            example['degraded_image'] = degraded_proj_range

        # cameras
        if self.condition_key == 'camera':
            cameras = self.load_camera(data_path)
            example[self.condition_key] = cameras

        return example


class NUSCENESDATABase(NUSCENESBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.condition_key in [None, 'image']  # for image input only

    def prepare_data(self):
        # read data paths from Nuscenes
        self.data = []
        if self.split == 'train':
            for seq_id in eval('NUSCENES_%s_SET' % self.split.upper()):
                self.data.extend(glob.glob(os.path.join(
                    self.data_root, f'v1.0-trainval{seq_id}_blobs_lidar/sweeps/LIDAR_TOP/*.bin')))
        elif self.split == 'val':
            self.data.extend(glob.glob(os.path.join(
                self.data_root, f'v1.0-test_blobs_lidar/samples/LIDAR_TOP/*.bin')))

class NUSCENESImageTrain(NUSCENESDATABase):
    def __init__(self, **kwargs):
        super().__init__(data_root='/media/ysx/mydata/nuscenes', split='train', **kwargs)


class NUSCENESImageValidation(NUSCENESDATABase):
    def __init__(self, **kwargs):
        super().__init__(data_root='/media/ysx/mydata/nuscenes', split='val', **kwargs)


        
