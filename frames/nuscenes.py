from nuscenes.nuscenes import NuScenes
from typing import Dict, List, Any, Optional


class NuScenesScene(object):
    scene: Optional[Dict[str, Any]]
    scene_number: int
    nusc: Optional[Any]

    def __init__(self, nusc, scene_number, config: Dict[str, Any]):
        self.scene = nusc.scene[scene_number]
