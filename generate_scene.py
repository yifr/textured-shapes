import os
import sys
import bpy
sys.path.append("/home/yyf/textured-shapes")

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
    favour_vec = np.random.random(3).tolist()  # which 3D dimension is preferred
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
    params["extrusions"] = int(amount)
    params["face_overlap"] = face_overlap
    params["rand_loc"] = rand_loc
    params["rand_rot"] = rand_rot
    params["rand_scale"] = rand_scale
    params["transform_seed"] = transform_seed
    params["is_subsurf"] = is_subsurf
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

def set_render_settings():
    # Set properties to increase speed of render time
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"  # use cycles for headless rendering
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512

    scene.render.image_settings.compression = 0
    scene.cycles.samples = 128

    scene.cycles.device = "GPU"

    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    device_type = "CUDA"
    cycles_preferences.compute_device_type = device_type

    activated_gpus = []
    print(cycles_preferences.get_devices())
    for device in cycles_preferences.devices:
        print("Activating: ", device)
        if not device.type == "CPU":
            device.use = True
            activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = device_type

def generate_passes(scene_dir):
    objects = bpy.data.objects
    scene = bpy.context.scene

    scene.use_nodes = True

    # Give each object in the scene a unique pass index
    scene.view_layers["ViewLayer"].use_pass_object_index = True
    scene.view_layers["ViewLayer"].use_pass_normal = True
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.view_layers["ViewLayer"].use_pass_mist = True

    for i, object in enumerate(objects):
        if object.name == "Plane":
            object.pass_index = 0
        else:
            object.pass_index = (i + 1)

    node_tree = scene.node_tree
    links = node_tree.links

    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    # Create a node for outputting the depth of each pixel from the camera
    depth_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    depth_output_node.name = "Depth_Output"
    depth_output_node.label = "Depth_Output"
    path = os.path.join(scene_dir, "depths")
    depth_output_node.base_path = path
    depth_output_node.location = 600, 0

    # Create a node for outputting the index of each object
    mask_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    mask_output_node.label = "Mask_Output"
    mask_output_node.name = "Mask_Output"
    mask_output_node.format.color_mode = "RGB"
    path = os.path.join(scene_dir, "masks")
    mask_output_node.base_path = path
    mask_output_node.location = 600, -200

    # Create a node for outputting the normal of each object
    normal_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    normal_output_node.label = "Normal_Output"
    normal_output_node.name = "Normal_Output"
    normal_output_node.format.file_format = "OPEN_EXR"
    normal_output_node.format.color_mode = "RGB"
    path = os.path.join(scene_dir, "normals")
    normal_output_node.base_path = path
    normal_output_node.location = 600, -300

    math_node = node_tree.nodes.new(type="CompositorNodeMath")
    math_node.operation = "DIVIDE"
    math_node.inputs[1].default_value = 255.0
    math_node.location = 400, -200

    depth_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    depth_output_node.label = "Depth_Output"
    depth_output_node.name = "Depth_Output"
    depth_output_node.format.file_format = "OPEN_EXR"
    depth_output_node.format.color_mode = "RGB"
    path = os.path.join(scene_dir, "depth")
    depth_output_node.base_path = path
    depth_output_node.location = 600, 0

    # Create a node for the output from the renderer
    compositor_node = node_tree.nodes.new(type="CompositorNodeComposite")
    compositor_node.location = 600, 200
    render_layers_node = node_tree.nodes.new(type="CompositorNodeRLayers")
    render_layers_node.location = -100, 0

    # Link all the nodes together
    links.new(render_layers_node.outputs["Image"], compositor_node.inputs["Image"])

    # Link Depth
    links.new(
        render_layers_node.outputs["Depth"], depth_output_node.inputs["Image"]
    )

    # Link Object Index Masks
    links.new(render_layers_node.outputs["IndexOB"], math_node.inputs[0])
    links.new(math_node.outputs[0], mask_output_node.inputs["Image"])

    # Link Normals
    links.new(
        render_layers_node.outputs["Normal"], normal_output_node.inputs["Image"]
    )

def render_scenes(scene_num):
    scene_path = f"/om2/user/yyf/textured-shapes/data/scene_{scene_num:03d}/"
    if not os.path.exists(scene_path):
        os.makedirs(scene_path)

    texture_configs = glob("TextureSamples/configs/*.json")
    obj_params, obj, background = setup_scene()

    # save object params
    with open(os.path.join(scene_path, "obj_params.json"), "w") as f:
        json.dump(obj_params, f)

    for i, texture_config in enumerate(texture_configs):
        texture_params = json.load(open(texture_config, "r"))
        texture_path = os.path.join(scene_path, f"texture_{i:02d}/")
        if not os.path.exists(texture_path):
            os.makedirs(texture_path)

        with open(os.path.join(texture_path, "texture_params.json"), "w") as f:
            json.dump(texture_params, f)

        foreground_texture = texture_params["foreground"]
        background_texture = texture_params["background"]
        materials.add_material(background_texture, background, "background")
        materials.add_material(foreground_texture, obj, "foreground")
        background.scale = (3, 3, 3)

        bpy.context.scene.render.filepath = texture_path
        bpy.ops.render.render(write_still=True, animation=True)

    # Render shaded + geometry passes
    materials.default_material(background, (0.8, 0.8, 0.9, 1))
    materials.default_material(obj, color="random") # random foreground color

    mask_filepath = os.path.join(scene_path, "render_passes/")
    generate_passes(mask_filepath)
    bpy.context.scene.render.filepath = os.path.join(scene_path, f"shaded/")
    bpy.ops.render.render(write_still=True, animation=True)

    # save object mesh
    bpy.context.view_layer.objects.active = obj
    obj_path = os.path.join(scene_path, "object.obj")
    bpy.ops.export_scene.obj(filepath=obj_path, use_selection=True)

if __name__=="__main__":
    set_render_settings()
    for i in range(50, 100):
        delete_all()
        render_scenes(i)
