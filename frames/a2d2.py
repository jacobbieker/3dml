import json
import numpy as np
import numpy.linalg as la
from typing import Dict, Any, Type, List, Sequence, Optional, Union
import cv2

"""

Audi A2D2 Utils

Primarily taken from the tutorial on their website for converting labels and points together

"""

EPSILON = 1.0e-10  # norm should not be small


def get_ego_view(config: Dict[str, Any]):
    return config['vehicle']['view']


def get_view(config: Dict[str, Any], camera_name: str):
    return config['cameras'][camera_name]['view']


def get_axes_of_a_view(view):
    x_axis = view['x-axis']
    y_axis = view['y-axis']

    x_axis_norm = la.norm(x_axis)
    y_axis_norm = la.norm(y_axis)

    if (x_axis_norm < EPSILON or y_axis_norm < EPSILON):
        raise ValueError("Norm of input vector(s) too small.")

    # normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm

    # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)

    # create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)

    # calculate and check y-axis and z-axis norms
    y_axis_norm = la.norm(y_axis)
    z_axis_norm = la.norm(z_axis)

    if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
        raise ValueError("Norm of view axis vector(s) too small.")

    # make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm

    return x_axis, y_axis, z_axis


def get_origin_of_a_view(view):
    return view['origin']


def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)

    # get origin
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)

    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis

    # origin
    transform_to_global[0:3, 3] = origin

    return transform_to_global


def get_transform_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    trans = np.eye(4)
    rot = np.transpose(transform_to_global[0:3, 0:3])
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])

    return trans


def get_rot_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    # get rotation
    rot = np.transpose(transform_to_global[0:3, 0:3])

    return rot


def get_rot_to_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    # get rotation
    rot = transform_to_global[0:3, 0:3]

    return rot


def rot_from_to(src, target):
    rot = np.dot(get_rot_from_global(target), get_rot_to_global(src))

    return rot


def transform_from_to(src, target):
    transform = np.dot(get_transform_from_global(target), \
                       get_transform_to_global(src))

    return transform


def project_lidar_from_to(lidar, src_view, target_view):
    lidar = dict(lidar)
    trans = transform_from_to(src_view, target_view)
    points = lidar['points']
    points_hom = np.ones((points.shape[0], 4))
    points_hom[:, 0:3] = points
    points_trans = (np.dot(trans, points_hom.T)).T
    lidar['points'] = points_trans[:, 0:3]

    return lidar


def load_and_transform_lidar_to_world(lidar_filename: Union[str], view, config) -> np.ndarray:
    lidar = np.load(lidar_filename)
    ego_view = get_ego_view(config)
    point_cloud = project_lidar_from_to(lidar, view, ego_view)
    return point_cloud


def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                      'camera_' + \
                      file_name_image[2] + '_' + \
                      file_name_image[3] + '.png'

    return file_name_image


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q


def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    image = np.copy(image_orig)

    # get rows and cols
    rows = (lidar['row'] + 0.5).astype(np.int)
    cols = (lidar['col'] + 0.5).astype(np.int)

    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['distance'])

    # get distances
    distances = lidar['distance']
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                                                np.sqrt(pixel_opacity), 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = \
            (1. - pixel_opacity) * \
            np.multiply(image[pixel_rows, pixel_cols, :], \
                        colours[i]) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)


def extract_semantic_file_name_from_image_file_name(file_name_image):
    file_name_semantic_label = file_name_image.split('/')
    file_name_semantic_label = file_name_semantic_label[-1].split('.')[0]
    file_name_semantic_label = file_name_semantic_label.split('_')
    file_name_semantic_label = file_name_semantic_label[0] + '_' + \
                               'label_' + \
                               file_name_semantic_label[2] + '_' + \
                               file_name_semantic_label[3] + '.png'

    return file_name_semantic_label


def skew_sym_matrix(u):
    return np.array([[0, -u[2], u[1]],
                     [u[2], 0, -u[0]],
                     [-u[1], u[0], 0]])


def axis_angle_to_rotation_mat(axis, angle):
    return np.cos(angle) * np.eye(3) + \
           np.sin(angle) * skew_sym_matrix(axis) + \
           (1 - np.cos(angle)) * np.outer(axis, axis)


def read_bounding_boxes(file_name_bboxes):
    # open the file
    with open(file_name_bboxes, 'r') as f:
        bboxes = json.load(f)

    boxes = []  # a list for containing bounding boxes
    print(bboxes.keys())

    for bbox in bboxes.keys():
        bbox_read = {}  # a dictionary for a given bounding box
        bbox_read['class'] = bboxes[bbox]['class']
        bbox_read['truncation'] = bboxes[bbox]['truncation']
        bbox_read['occlusion'] = bboxes[bbox]['occlusion']
        bbox_read['alpha'] = bboxes[bbox]['alpha']
        bbox_read['top'] = bboxes[bbox]['2d_bbox'][0]
        bbox_read['left'] = bboxes[bbox]['2d_bbox'][1]
        bbox_read['bottom'] = bboxes[bbox]['2d_bbox'][2]
        bbox_read['right'] = bboxes[bbox]['2d_bbox'][3]
        bbox_read['center'] = np.array(bboxes[bbox]['center'])
        bbox_read['size'] = np.array(bboxes[bbox]['size'])
        angle = bboxes[bbox]['rot_angle']
        axis = np.array(bboxes[bbox]['axis'])
        bbox_read['rotation'] = axis_angle_to_rotation_mat(axis, angle)
        boxes.append(bbox_read)

    return boxes


def extract_bboxes_file_name_from_image_file_name(file_name_image):
    file_name_bboxes = file_name_image.split('/')
    file_name_bboxes = file_name_bboxes[-1].split('.')[0]
    file_name_bboxes = file_name_bboxes.split('_')
    file_name_bboxes = file_name_bboxes[0] + '_' + \
                       'label3D_' + \
                       file_name_bboxes[2] + '_' + \
                       file_name_bboxes[3] + '.json'

    return file_name_bboxes


def get_points(bbox):
    half_size = bbox['size'] / 2.

    if half_size[0] > 0:
        # calculate unrotated corner point offsets relative to center
        brl = np.asarray([-half_size[0], +half_size[1], -half_size[2]])
        bfl = np.asarray([+half_size[0], +half_size[1], -half_size[2]])
        bfr = np.asarray([+half_size[0], -half_size[1], -half_size[2]])
        brr = np.asarray([-half_size[0], -half_size[1], -half_size[2]])
        trl = np.asarray([-half_size[0], +half_size[1], +half_size[2]])
        tfl = np.asarray([+half_size[0], +half_size[1], +half_size[2]])
        tfr = np.asarray([+half_size[0], -half_size[1], +half_size[2]])
        trr = np.asarray([-half_size[0], -half_size[1], +half_size[2]])

        # rotate points
        points = np.asarray([brl, bfl, bfr, brr, trl, tfl, tfr, trr])
        points = np.dot(points, bbox['rotation'].T)

        # add center position
        points = points + bbox['center']

    return points

