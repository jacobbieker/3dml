from nuscenes.nuscenes import NuScenes
from typing import Dict, List, Any, Optional, Union
from .scene import Scene


class NuScenesScene(Scene):

    def __init__(self, nusc, scene_number, config: Dict[str, Any]):
        super().__init__()
        self.loader = nusc[scene_number]
        self.nusc = nusc
        sample_token = self.loader["first_sample_token"]
        sample = nusc.get("sample", sample_token)
        i = 0
        while sample_token != "":
            i += 1
            sample_token = sample["next"]
        self.num_frames = i

    def convert(self):
        pass

    def stack_frames(self, frame_numbers: List[int]):
        pass

    def get_frames(self, frame_numbers: List[int], stack: bool = False):
        pass

    def load_metadata(self):
        pass

    def load_data(self, frame_numbers: Union[List[int], int]):
        frame_imgs = []
        frame_points = []
        frame_objects = []
        frame_data = {}
        frame_no_to_full_frame_no = {}
        full_frame_no_to_frame_no = {}
        if frame_numbers is None:
            frame_numbers = list(range(self.num_frames))
        if isinstance(frame_numbers, int):
            frame_numbers = [frame_numbers]

        sample_token = self.loader["first_sample_token"]
        scene_token = self.loader["token"]
        log_token = self.loader["log_token"]
        sample = self.nusc.get("sample", sample_token)
        for i, frame_no in enumerate(frame_numbers):
            if i not in frame_numbers:
                continue

