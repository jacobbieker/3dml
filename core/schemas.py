from schema import Schema, And, Optional, Use
import numpy as np

"""

Common Schemas to transform the various formats into

"""

CUBOID_SCHEMA = Schema({
    "xyz": None,
    "wlh": None,
    "label": None,
    "attributes": None,
    "yaw": None,
    Optional("points"): None,
    Optional("visible_cameras"): None,
})

GT_CUBOIDS_SCHEMA = [CUBOID_SCHEMA]

ANNOTATION_SCHEMA = Schema({
    Optional("points"): None,
    Optional("reflectance"): None,
    "device": None, # Ego Vehicle location in global coordinates
    Optional("cameras"): None,
    Optional("attributes"): None,
    "cuboids": GT_CUBOIDS_SCHEMA,
    "coordinate_system": lambda x: x in ["ego", "world"]
})

SEGMENTATION_SCHEMA = Schema({
    Optional("points"): None,
    Optional("reflectance"): None,
    "device": None, # Ego Vehicle location in global coordinates
    Optional("cameras"): None,
    Optional("attributes"): None,
    "mask": None,
    "coordinate_system": lambda x: x in ["ego", "world"]
})



