import numpy as np
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import cv2


class Camera(ABC):
    distortion: np.ndarray
    matrix: np.ndarray
    orig_matrix: Optional[np.ndarray]
    width: int
    height: int
    image: Optional[np.ndarray]
    lens: str
    index: int
    name: str

    @abstractmethod
    def load_image(self, config: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def load_metadata(self, config: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def project(self, points: np.ndarray):
        """ Project points/bounding box, etc. to camera plane"""
        raise NotImplementedError


@dataclass
class A2D2Camera(Camera):
    def load_image(self, config: Dict[str, Any]):
        pass

    def load_metadata(self, config: Dict[str, Any]):
        pass

    def project(self, points: np.ndarray):
        if self.name in ['front_left', 'front_center', \
                         'front_right', 'side_left', \
                         'side_right', 'rear_center']:
            if self.lens == 'fisheye':
                return cv2.fisheye.undistortImage(points, self.orig_matrix, \
                                                  D=self.distortion, Knew=self.matrix)
            elif self.lens == 'telecam':
                return cv2.undistort(points, self.orig_matrix, \
                                     distCoeffs=self.distortion, newCameraMatrix=self.matrix)
            else:
                return points
        else:
            return points
