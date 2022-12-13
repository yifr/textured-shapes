import os
import sys
import bpy
sys.path.append("/Users/yonifriedman/Projects/GestaltNeurophys")

import util
import materials
import json
import pickle
import numpy as np
import pyquaternion
from glob import glob

"""
Render a scene with a textured foreground object and a textured background plane
Rotate the camera in a jacobean spiral, and parent the background plane to the camera
so it looks like the object is moving.

Save:
1. The texture parameters
2. The object parameters (shape parameters and rotation?)
3. The camera extrinsics for each frame and camera intrinsics
4. Depth maps for each frame
5. Surface normals for each frame
6. The rendered images 
"""

def create_object():
    params = {}
    seed = np.random.randint(0, 10000000)
    mirror_x = 0
    mirror_y = 0
    mirror_z = 0
    n_big = 1   # n big/medium/small shapes that get concatenated
    n_med = 0
    n_small = 0
    favour_vec = np.random.random(3)  # which 3D dimension is preferred
    amount = np.random.randint(5,10)  # number of extrusion
    face_overlap = 1
    rand_loc = 0
    rand_rot = 1
    rand_scale = 0
    transform_seed = 0
    is_subsurf = 1 # whether you do subsurface divisions that smooth it out. 1 will make it smooth
    subsurf_subdivisions = 5
    is_bevel = np.random.choice([0, 1]) # bevels it

    params["seed"] = seed
    params["mirror_x"] = mirror_x
    params["mirror_y"] = mirror_y
    params["mirror_z"] = mirror_z
    params["n_big"] = n_big
    params["n_med"] = n_med
    params["n_small"] = n_small
    params["favour_vec"] = favour_vec
    params["extrusions"] = amount
    params["face_overlap"] = face_overlap
    params["rand_loc"] = rand_loc
    params["rand_rot"] = rand_rot
    params["rand_scale"] = rand_scale
    params["transform_seed"] = transform_seed
    params["is_subsurf"] = is_subsurf
    params["subsurf_subdivisions"] = subsurf_subdivisions
    params["is_bevel"] = is_bevel

    bpy.ops.mesh.shape_generator(
        random_seed=seed,
        mirror_x=int(mirror_x),
        mirror_y=int(mirror_y),
        mirror_z=int(mirror_z),
        big_shape_num=int(n_big),
        medium_shape_num=int(n_med),
        small_shape_num=int(n_small),
        favour_vec=favour_vec,
        amount=int(amount),
        prevent_ovelapping_faces=int(face_overlap),
        randomize_location=int(rand_loc),
        randomize_rotation=int(rand_rot),
        randomize_scale=int(rand_scale),
        random_transform_seed=int(transform_seed),
        is_subsurf=int(is_subsurf),
        subsurf_subdivisions=int(subsurf_subdivisions),
        is_bevel=int(is_bevel),
    )
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
    bpy.ops.object.join()

    return params

def setup_scene(scene_num=0):
    # Add background
    bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=(0, 0, -3), scale=(1, 1, 1))
    background = bpy.context.object
    
    # Add foreground object
    object_params = create_object()
    obj = bpy.context.object
    obj.rotation_euler = np.random.random(3) * 2 * np.pi

    # add light
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.object
    sun.data.use_shadow = False
    sun.data.specular_factor = 0.
    sun.data.energy = 1.
    
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 8), rotation=(0, 0, 0))
    bpy.context.scene.camera = bpy.context.object
    camera = bpy.context.object
    
    bpy.data.objects["Plane"].parent = camera
    bpy.data.objects["Plane"].matrix_parent_inverse = camera.matrix_world.inverted()
    
    sphere_radius = 10
    num_observations = 72
    bpy.context.scene.frame_end = num_observations
    cam_locations = util.get_archimedean_spiral(sphere_radius, num_observations)
    obj_location = np.zeros((1,3))

    cv_poses = util.look_at(cam_locations, obj_location)
    blender_poses = [util.cv_cam2world_to_bcam2world(m) for m in cv_poses]
    for i, pose in enumerate(blender_poses):
        camera.matrix_world = pose
        # insert keyframe for location and rotation
        camera.keyframe_insert(data_path="location", frame=i)
        camera.keyframe_insert(data_path="rotation_euler", frame=i)

    camera.data.lens = 100

    return object_params, obj, background

def delete_all():
    bpy.ops.object.select_all(action="DESELECT")
    objs = bpy.data.objects
    for obj in objs:
        objs.remove(obj, do_unlink=True)


def render_scenes(scene_num):
    texture_configs = glob("/Users/yonifriedman/Projects/GestaltNeurophys/TextureSamples/configs/*.json")
    obj_params, obj, background = setup_scene()

    for i, texture_config in enumerate(texture_configs):
        texture_params = json.load(open(texture_config, "r"))
        
        foreground_texture = texture_params["foreground"]
        background_texture = texture_params["background"]
        materials.add_material(background_texture, background, "background")
        materials.add_material(foreground_texture, obj, "foreground")
        background.scale = (3, 3, 3)

        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512
        bpy.context.scene.render.filepath = f"/Users/yonifriedman/Projects/GestaltNeurophys/data/scene_{scene_num:03d}/texture_{i:02d}/"
        bpy.ops.render.render(write_still=True, animation=True)

for i in range(100):
    delete_all()
    render_scenes(i)