import random
import bpy
from mathutils import Matrix, Vector
import os
import numpy as np
import math
from functools import reduce

#  VS Code bug workaround for "unreachable code" bug using np.cross
# see: https://github.com/microsoft/pylance-release/issues/3277
cross1 = lambda x,y:np.cross(x,y)

def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)

# All the following functions follow the opencv convention for camera coordinates.
def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0., -1., 0.])

    right = cross1(tmp, forward)
    right = normalize(right)

    up = cross1(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0., 0., 0., 1.]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def look_atxy(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0., -1., 0.])

    right = cross1(tmp, forward)
    right = normalize(right)

    up = cross1(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0., 0., 0., 1.]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat

def sample_spherical(n, radius=1.):
    xyz = np.random.normal(size=(n,3))
    xyz = normalize(xyz) * radius
    return xyz

def sample_yaw(n, radius=1.):
    xyz = np.random.normal(size=(n,3))
    xyz[:,1] = 0
    xyz = normalize(xyz) * radius

    return xyz

def sample_xy(n, trans):
    x = np.random.uniform(-trans,trans, size=(n,1))
    y = np.random.uniform(-trans,trans, size=(n,1))
    z = np.zeros((n,1))
    xyz = np.concatenate((x,y,z), axis=1)
    return xyz

def sample_controlled_yaw(n, radius=1.):
    xyz = np.zeros(shape=(n,3))
    stepsize = 2/n*np.pi
    xyz[:,1] = np.sin(45 * math.pi / 180)
    for i in range(n):
        xyz[i,0] = np.sin(i*stepsize)
        xyz[i,2] = np.cos(i*stepsize)
    xyz = normalize(xyz) * radius

    return xyz


def set_camera_focal_length_in_world_units(camera_data, focal_length):
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera_data.sensor_width
    sensor_height_in_mm = camera_data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camera_data.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    camera_data.lens = focal_length / s_u


# Blender: camera looks in negative z-direction, y points up, x points right.
# Opencv: camera looks in positive z-direction, y points down, x points right.
def cv_cam2world_to_bcam2world(cv_cam2world):
    '''

    :cv_cam2world: numpy array.
    :return:
    '''
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    cam_location = Vector(cv_cam2world[:3, -1].tolist())
    cv_cam2world_rot = Matrix(cv_cam2world[:3, :3].tolist())

    cv_world2cam_rot = cv_cam2world_rot.transposed()
    cv_translation = -1. * cv_world2cam_rot @ cam_location

    blender_world2cam_rot = R_bcam2cv @ cv_world2cam_rot
    blender_translation = R_bcam2cv @ cv_translation

    blender_cam2world_rot = blender_world2cam_rot.transposed()
    blender_cam_location = -1. * blender_cam2world_rot @ blender_translation

    blender_matrix_world = Matrix((
        blender_cam2world_rot[0][:] + (blender_cam_location[0],),
        blender_cam2world_rot[1][:] + (blender_cam_location[1],),
        blender_cam2world_rot[2][:] + (blender_cam_location[2],),
        (0, 0, 0, 1)
    ))

    return blender_matrix_world


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_world2cam_from_blender_cam(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2] # Matrix_world returns the cam2world matrix.
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],),
        (0,0,0,1)
    ))
    return RT


#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm


    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
         (    0  , alpha_v, v_0),
         (    0  , 0,        1 )))
    return K


def cond_mkdir(path):
    path = os.path.normpath(path)
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def dump(obj):
    for attr in dir(obj):
        if hasattr(obj, attr):
            print("obj.%s = %s" % (attr, getattr(obj, attr)))

def get_archimedean_spiral(sphere_radius, num_steps=250):
    '''
    https://en.wikipedia.org/wiki/Spiral, section "Spherical spiral". c = a / pi
    '''
    a = 40
    r = sphere_radius

    translations = []

    i = a / 2
    while i < a:
        theta = i / a * math.pi
        x = r * math.sin(theta) * math.cos(-i)
        z = r * math.sin(-theta + math.pi) * math.sin(-i)
        y = r * - math.cos(theta)

        translations.append((x, y, z))
        i += a / (2 * num_steps)

    return np.array(translations)


def blender_mat_from_euler_translation(euler_angles, translation):
    hom_coords = np.array([[0., 0., 0., 1.]]).reshape(1, 4)
    rot_mat = euler2mat(euler_angles[0], euler_angles[1], euler_angles[2])
    obj_pose_cv = np.concatenate((rot_mat, translation.reshape(3,1)), axis=-1)
    obj_pose_cv = np.concatenate((obj_pose_cv, hom_coords), axis=0)
    return cv_cam2world_to_bcam2world(obj_pose_cv)


def euler2mat(x=0, y=0, z=0):
    ''' Return matrix for rotations around z, y and x axes

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles

    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True

    The output rotation matrix is equal to the composition of the
    individual rotations

    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True

    You can specify rotations by named arguments

    >>> np.all(M3 == euler2mat(x=xrot))
    True

    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.

    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)

    Rotations are counter-clockwise.

    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True

    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def get_specific_views(sphere_radius):
    '''
    custom views test_specific_views
    '''
    r = sphere_radius
    a = 1/math.sqrt(3)
    translations = [(r, 0, 0), (0.0001, r*-0.999, 0.0001),
                    (a*r, a*r, a*-r), (a*-r, a*r, a*r)]
    return np.array(translations)


def get_thomas_views(sphere_radius,n):
    translations = [(r, 0, 0), (0.0001, r*-0.999, 0.0001),
                    (a*r, a*r, a*-r), (a*-r, a*r, a*r)]
    return np.array(translations)
