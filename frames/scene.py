from typing import List, Dict, Any, Optional, Union
from ..core.schemas import ANNOTATION_SCHEMA, SEGMENTATION_SCHEMA
from abc import ABC, abstractmethod
import numpy as np


class Scene(ABC):

    def __init__(self):
        self.frames = Optional[Dict]
        self.cameras = {}
        self.lidars = {}
        self.radars = {}
        self.metadata = {}
        self.frame_data = {}
        self.loader = None
        self.num_frames = Optional[int]
        self.include_annotations = None
        self.frame_no_to_full_frame_no = {}
        self.full_frame_no_to_frame_no = {}

    @abstractmethod
    def convert(self):
        """Convert from different formats to common format"""
        raise NotImplementedError

    @abstractmethod
    def stack_frames(self, frame_numbers: List[int]):
        raise NotImplementedError

    def get_frames(self, frame_numbers: List[int], stack: bool = False):
        raise NotImplementedError

    def load_metadata(self):
        raise NotImplementedError

    def load_data(self, frame_numbers: Union[List[int], int]):
        raise NotImplementedError
