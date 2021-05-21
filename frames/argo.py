import argoverse
from argoverse.utils.camera_stats import RING_CAMERA_LIST, CAMERA_LIST
from argoverse.sensor_dataset_config import ArgoverseConfig
from argoverse.utils.se3 import SE3
from argoverse.utils.calibration import CameraConfig, get_calibration_config, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_image_dims_for_camera, point_cloud_to_homogeneous, project_lidar_to_img_motion_compensated, project_lidar_to_undistorted_img, project_lidar_to_img
from typing import Dict, List, Any, Optional, Union
from .scene import Scene
import numpy as np


class ArgoVerseScene(Scene):

    def __init__(self, argo, cameras: Optional[List[int]]):
        super().__init__()
        self.loader = argo
        self.num_frames = self.loader.lidar_count # Only care about the key frames for now
        if cameras is None:
            cameras = list(range(len(RING_CAMERA_LIST)))
        for i, cam in enumerate(RING_CAMERA_LIST):
            if i in cameras:
                self.cameras[i] = cam

    def convert(self):
        pass

    def stack_frames(self, frame_numbers: List[int]):
        pass

    def get_frames(self, frame_numbers: List[int], stack: bool = False):
        if stack:
            return self.stack_frames(frame_numbers)
        for frame_no in frame_numbers:
            return self.frame_data[self.frame_no_to_full_frame_no[frame_no]]

    def load_metadata(self):
        pass

    def load_data(self, frame_numbers: Union[List[int], int]):
        frame_imgs = []
        frame_pointcloud = []
        frame_objects = []
        frame_data = {}
        frame_no_to_full_frame_no = {}
        full_frame_no_to_frame_no = {}
        if isinstance(frame_numbers, int):
            frame_numbers = [frame_numbers]

        for i, frame_no in enumerate(frame_numbers):
            cam_imgs = []
            for idx, cam in self.cameras.items():
                cam_imgs.append(self.loader.get_image_sync(frame_no, camera=cam))
            point_cloud = self.loader.get_lidar(frame_no)
            gt_objects = self.loader.get_label_object(frame_no)

            frame_imgs.append(cam_imgs)
            frame_pointcloud.append(point_cloud)
            frame_objects.append(gt_objects)
            frame_data[frame_no] = {"images": np.asarray(cam_imgs), "points": point_cloud, "objects": gt_objects}
            frame_no_to_full_frame_no[i] = frame_no
            full_frame_no_to_frame_no[frame_no] = i

        self.frame_data = frame_data
        self.frame_no_to_full_frame_no = frame_no_to_full_frame_no
        self.full_frame_no_to_frame_no = full_frame_no_to_frame_no

