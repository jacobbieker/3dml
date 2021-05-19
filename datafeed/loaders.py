from dataclasses import dataclass
from abc import ABC, abstractmethod
from schema import Schema
from typing import Dict, Any, Type, List, Sequence, Optional, Union
import numpy as np
import json

"""

Methods for loadings the various open 3D AV datasets into a common format for use throughout this repo

This is setup for the Audi a2d2, Waymo Open Dataset, NuScenes, Argoverse, ApolloScape, Bosch's Boxy Dataset, and Pandaset

Attempts to load the different modules for loading in the datasets, but should just pass if the setup doesn't exist

The loader should get the data from the native format for that loader, transform it to a common label/ format, then 
return that to the caller. 

"""

from abc import ABC


class Loader(ABC):
    point_cloud: bool
    segmentation: bool
    bounding_boxes: bool
    cameras: List[int]
    bus: bool

    def __init__(self, config: Dict[str, Any]):
        self.point_cloud = config.get("include_points", False)
        self.cameras = config.get("include_cameras", [])
        self.segmentation = config.get("include_segmentation", False)
        self.bounding_boxes = config.get("include_bboxes", False)
        self.bux = config.get("include_bus", False)

    @property
    def name(self) -> str:
        return type(self).__name__

    def __call__(self, scene_id: Any, frame_numbers: Union[int, List[int]]):
        return self.transform(self.get(scene_id, frame_numbers)

    @abstractmethod
    def get(self, scene_id: Any, frame_numbers: Union[int, List[int]]):
        """Gets an example, or set of examples """
        raise NotImplementedError

    @abstractmethod
    def transform(self, items: Any) -> Any:
        """ Convert the gathered annotations/frames to the common format"""
        raise NotImplementedError


class NuScenesLoader(Loader):
    """NuScenes has lidar segmentation for some frames, images, bounding boxes"""

    def get(self, scene_id: Any, frame_numbers: Union[int, List[int]]):
        return NotImplementedError

    def transform(self, items: Any) -> Any:
        return NotImplementedError


class A2D2Loader(Loader):
    """A2D2 has lidar segmentation of cameras, and bounding boxes"""

    def __init__(self, config: Dict[str, Any]):
        self.super.__init__(config)
        # Specific configuration file for A2D2
        self.config = json.load(open(config.get("a2d2_config", ""), 'r'))

    def get(self, scene_id: Any, frame_numbers: Union[int, List[int]]):

        return NotImplementedError

    def transform(self, items: Any) -> Any:
        return NotImplementedError


class ApolloScapeLoader(Loader):
    """"ApolloScape has stereo images, cameras, lidar, semantic segmentation of images"""

    def get(self, scene_id: Any, frame_numbers: Union[int, List[int]]):
        return NotImplementedError

    def transform(self, items: Any) -> Any:
        return NotImplementedError


class ArgoverseLoader(Loader):
    """"The Argoverse data includes lidar, images, and bounding boxes, as well as high quality maps"""

    def get(self, scene_id: Any, frame_numbers: Union[int, List[int]]):
        return NotImplementedError

    def transform(self, items: Any) -> Any:
        return NotImplementedError


class WaymoLoader(Loader):
    """" Waymo Open Dataset includes Lidar and camera information, with both 2D and 3D boudning boxes"""

    def get(self, scene_id: Any, frame_numbers: Union[int, List[int]]):
        return NotImplementedError

    def transform(self, items: Any) -> Any:
        return NotImplementedError


class BoxyLoader(Loader):
    """ Bosch Boxy Dataset is only camera images and 2D bounding boxes with 5MP images, and ~2million vehicles"""

    def get(self, scene_id: Any, frame_numbers: Union[int, List[int]]):
        return NotImplementedError

    def transform(self, items: Any) -> Any:
        return NotImplementedError
