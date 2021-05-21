from nuscenes.nuscenes import NuScenes
from typing import Dict, List, Any, Optional
from .scene import Scene


class NuScenesScene(Scene):

    def __init__(self, nusc, scene_number, config: Dict[str, Any]):
        super().__init__()
        self.loader = nusc.scene[scene_number]

    def convert(self):
        pass

    def stack_frames(self, frame_numbers: List[int]):
        pass

    def get_frames(self, frame_numbers: List[int], stack: bool = False):
        pass

    def load_metadata(self):
        pass
