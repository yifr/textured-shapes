import numpy as np
import glob
import json
import bpy
import os

def create_shape():
    params = {}
    seed = np.random.randint(0, 10000)
    mirror_x = 0
    mirror_y = 0
    mirror_z = 0
    n_big = 1   # n big/medium/small shapes that get concatenated
    n_med = 0
    n_small = 0
    favour_vec = np.random.random(3).tolist()  # which 3D dimension is preferred
    amount = np.random.randint(5,10)  # number of extrusion
    face_overlap = 1
    rand_loc = 0
    rand_rot = 1
    rand_scale = 0
    transform_seed = 0
    is_subsurf = np.random.choice([0, 1]) # whether you do subsurface divisions that smooth it out. 1 will make it smooth
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
    params["extrusions"] = int(amount)
    params["face_overlap"] = face_overlap
    params["rand_loc"] = rand_loc
    params["rand_rot"] = rand_rot
    params["rand_scale"] = rand_scale
    params["transform_seed"] = transform_seed
    params["is_subsurf"] = int(is_subsurf)
    params["subsurf_subdivisions"] = subsurf_subdivisions
    params["is_bevel"] = int(is_bevel)

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
    bpy.context.object.scale = (0.3, 0.3, 0.3)

    return params

def load_shape(params):
    seed = params["seed"]
    mirror_x = params["mirror_x"]
    mirror_y = params["mirror_y"]
    mirror_z = params["mirror_z"]
    n_big = params["n_big"]
    n_med = params["n_med"]
    n_small = params["n_small"]
    favour_vec = params["favour_vec"]
    amount = params["extrusions"]
    face_overlap = params["face_overlap"]
    rand_loc = params["rand_loc"]
    rand_rot = params["rand_rot"]
    rand_scale = params["rand_scale"]
    transform_seed = params["transform_seed"]
    is_subsurf = params["is_subsurf"]
    subsurf_subdivisions = params["subsurf_subdivisions"]
    is_bevel = params["is_bevel"]

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
    obj = bpy.context.active_object
    if "rotation_euler" in params.keys():
        re = params["rotation_euler"]
        obj.rotation_euler.x = re.x
        obj.rotation_euler.y = re.y
        obj.rotation_euler.z = re.z

    bpy.context.object.scale = (0.3, 0.3, 0.3)

def sample_shapenet_obj(obj_file=None):
    if obj_file is not None:
        bpy.ops.import_scene.obj(filepath=obj_file)
        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
        return obj_file

    seed = 42
    shapenet_dir = "/om/data/public/ShapeNet/ShapeNetCore.v2/"
    taxonomy_opts = [str(f.path) for f in os.scandir(shapenet_dir) if f.is_dir()]
    synset_opt = np.random.choice(taxonomy_opts)
    synset_dir = os.path.join(shapenet_dir, synset_opt)
    synset_shape = np.random.choice(glob.glob(synset_dir + "/*"))
    shapenet_obj = os.path.join(synset_dir, synset_shape, "models/model_normalized.obj")

    # deselect other objects
    bpy.ops.object.select_all(action='DESELECT')

    # import shapenet object
    bpy.ops.import_scene.obj(filepath=shapenet_obj)
    bpy.ops.object.join()
    bpy.ops.transform.resize(value=(3, 3, 3))
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')

    return shapenet_obj

