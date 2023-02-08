import json
import bpy 

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