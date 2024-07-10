import bpy
import numpy as np
import sys
sys.path.append("/users/yonifriedman/projects/textured-shapes/")
import icp
import shape_utils
from scipy.spatial.distance import cdist

def get_points(obj):
    vertices = obj.data.vertices
    points = []
    for vertex in vertices:
        points.append([vertex.co.x, vertex.co.y, vertex.co.z])
    return np.array(points)


def select_closest_points(array1, array2):
    """
    Select the closest points from array2 to array1 based on L2 distance.
    
    :param array1: numpy array of shape (M, 3)
    :param array2: numpy array of shape (N, 3), where N > M
    :return: numpy array of shape (M, 3) containing the M closest points from array2
    """
    if array1.shape[0] == array2.shape[0]:
        return array1, array2
    
    # Ensure array1 is the smaller array
    if array1.shape[0] > array2.shape[0]:
        array1, array2 = array2, array1
    
    # Calculate pairwise distances between all points
    distances = cdist(array1, array2)
    
    # For each point in array1, find the index of the closest point in array2
    closest_indices = np.argmin(distances, axis=1)
    
    # Select the closest points from array2
    closest_points = array2[closest_indices]
    
    return array1, closest_points


def test():
    shape_1 = shape_utils.create_shape()
    obj_1 = bpy.context.active_object
    shape_2 = shape_utils.create_shape()
    obj_2 = bpy.context.active_object
    
    points_1 = get_points(obj_1)
    points_2 = get_points(obj_2)
    
    p1_matched, p2_matched = select_closest_points(points_1, points_2)
    transform, distances, iterations = icp.icp(p1_matched, p2_matched)
    
    diff = np.mean(distances)
    return diff

diff1 = test()

print(diff1)
