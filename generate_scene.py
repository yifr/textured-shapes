import os
import sys
import bpy
print(os.getcwd())
sys.path.append(os.getcwd())
import itertools
import shape_utils
import util
import datetime
import materials
import json
import pickle
import numpy as np
import pyquaternion
from glob import glob
from BlenderArgparse import ArgParser


parser = ArgParser()
parser.add_argument("--start_scene", type=int, default=0)
parser.add_argument("--n_scenes", type=int, default=1)
parser.add_argument("--data_dir", type=str, default="/om2/user/yyf/textured-shapes/scenes")
parser.add_argument("--shape_type", type=str, default="shapegen")
parser.add_argument("--check_existing_obj", action="store_true")
parser.add_argument("--render_pass_only", action="store_true")
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--no_overwrite", action="store_true")
parser.add_argument("--no_render", action="store_true")
parser.add_argument("--num_trajectories", type=int, default=1, help="number of camera trajectories to render")
parser.add_argument("--max_trajectories", type=int, default=10, help="number of camera trajectories to sample from")
parser.add_argument("--textures_per_scene", type=int, default=1, help="How many textures to render for a given shape")
parser.add_argument("--num_frames", type=int, default=90, help="How many frames to render")
parser.add_argument("--device_type", type=str, default="CUDA", help="CUDA | METAL | CPU")
parser.add_argument("--focal_length", type=str, default="large", help="large (132.2 mm) | medium (.6 * large)| small (.2 * large)")
parser.add_argument("--save_blendfile", action="store_true", help="Save the blend file for each scene")
parser.add_argument("--alternate_obj_directory", type=str, default="", help="Directory where corresponding obj files are stored")
args = parser.parse_args()


def load_object(scene_path, args):
    # Add foreground object
    if args.alternate_obj_directory != "":
        scene_id = scene_path.split("/")[-1]
        obj_path = os.path.join(args.alternate_obj_directory, scene_id, "shape.obj")
        #bpy.ops.import_scene.obj(filepath=obj_path)

        obj_params_file = os.path.join(args.alternate_obj_directory, scene_id, 'obj_params_updated.json')
        object_params = json.load(open(obj_params_file, 'rb'))
        shape_utils.load_shape(object_params)
        obj = bpy.context.active_object
        obj.name = "Object"
        obj = bpy.data.objects[-1]
        obj.name = "Object"

    else:
        obj_params_file = os.path.join(scene_path, 'obj_params.json')
        if os.path.exists(obj_params_file) and args.check_existing_obj:
            if args.shape_type == "shapenet":
                obj_params = obj_params_file['params']
                shape_utils.sample_shapenet_obj(obj_params)
            else:
                object_params = json.load(open(obj_params_file, 'rb'))
                shape_utils.load_shape(object_params)
        else:
            if args.shape_type == "shapenet":
                object_file = shape_utils.sample_shapenet_obj()
                object_params = {"params": object_file}
            else:
                object_params = shape_utils.create_shape()

            with open(obj_params_file, "w") as f:
                json.dump(object_params, f)

    if args.shape_type == "shapenet":
        obj = bpy.data.objects["model_normalized"]
    else:
        obj = bpy.data.objects[-1]

    obj.name = "Object"
    return object_params, obj


def load_pose(filename):
    lines = open(filename, "r").read().splitlines()
    pose = np.zeros((4, 4), dtype=np.float32)
    for i in range(16):
        pose[i // 4, i % 4] = lines[0].split(" ")[i]
    return pose.squeeze()


def sample_and_set_cam_poses(pose_id, num_observations, sphere_radius, all_poses_dir="poses"):
    """
    Chooses trajectory, sets camera poses and saves them to a folder (generates new trajectories if folder is not initialized with positions)
    Params:
        scene_path: path to scene (object + texture combo) where cam poses will be saved
        num_observations: number of steps to generate in trajectory (each one will be rendered out as a keyframe)
        trajectory_radius: assumes object is centered in world coordinates -- will generate a trajectory with radius around center
        all_poses_dir: where to look for existing trajectories if sampling is turned on
        save_limit: Max number of trajectories expected in save path -- if there are fewer than that, a new one will be generated and saved

    Returns:
        ID of sampled trajectory
    """

    camera = bpy.context.scene.camera
    bpy.context.scene.frame_end = num_observations

    all_poses_dir = os.path.join(os.getcwd(), all_poses_dir)
    if not os.path.exists(all_poses_dir):
        os.makedirs(all_poses_dir)

    pose_dir = os.path.join(all_poses_dir, f"pose_{pose_id}")

    blender_poses = []

    if not os.path.exists(pose_dir):
        cam_locations = util.sample_controlled_yaw(num_observations, sphere_radius)    #util.sample_trajectory(num_observations, sphere_radius)
        obj_location = np.zeros((1,3))

        cv_poses = util.look_atxy(cam_locations, obj_location)
        blender_poses = [util.cv_cam2world_to_bcam2world(m) for m in cv_poses]
        os.makedirs(pose_dir)

        # Save poses
        for i, pose in enumerate(blender_poses):
            # insert keyframe for location and rotation
            camera.matrix_world = pose
            camera.keyframe_insert(data_path="location", frame=i+1)
            camera.keyframe_insert(data_path="rotation_euler", frame=i+1)

            RT = util.get_world2cam_from_blender_cam(camera)
            cam2world = RT.inverted()

            # Write out camera pose
            with open(os.path.join(pose_dir, f'{(i+1):06d}.txt'),'w') as pose_file:
                matrix_flat = []
                for j in range(4):
                    for k in range(4):
                        matrix_flat.append(cam2world[j][k])
                pose_file.write(' '.join(map(str, matrix_flat)) + '\n')
    else:
        # Sample and read poses
        for i in range(num_observations):
            pose_file = os.path.join(pose_dir, f"{(i+1):06d}.txt")
            pose = load_pose(pose_file)

            pose = util.cv_cam2world_to_bcam2world(pose) # pose is saved in world coordinates, and we want it in camera coordinates

            # insert keyframe for location and rotation
            camera.matrix_world = pose
            camera.keyframe_insert(data_path="location", frame=i+1)
            camera.keyframe_insert(data_path="rotation_euler", frame=i+1)

    return pose_id, pose_dir


def setup_ecological_scene_geometry(scene_path="", args=None):
    """
    In this kind of setting, the object is surrounded by walls and the camera moves freely around it.
    """

    bpy.context.scene.use_nodes = False

    # Get the environment node tree of the current scene
    world_node_tree = bpy.context.scene.world.node_tree
    tree_nodes = world_node_tree.nodes

    # Clear all nodes
    tree_nodes.clear()

    # Add Background node
    node_background = tree_nodes.new(type='ShaderNodeBackground')

    # Add Output node
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = 200,0

    # Link all nodes
    links = world_node_tree.links
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

    # Add background
    def add_and_name_wall(location, rotation, name, size=10):
        bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=location, rotation=rotation)
        bpy.context.selected_objects[0].name = name

    wall_scale = 10
    init_size = 1
    offset = wall_scale / 2
    add_and_name_wall((0, 0, -offset), (0, 0, 0), "left_wall_z", size=init_size)
    add_and_name_wall((0, 0, offset), (0, 0, 0), "right_wall_z", size=init_size)

    add_and_name_wall((-offset, 0, 0), (0, np.pi/2, 0), "left_wall_x", size=init_size)
    add_and_name_wall((offset, 0, 0), (0, np.pi/2, 0), "right_wall_x", size=init_size)

    add_and_name_wall((0, -offset, 0), (np.pi/2, 0, 0), "left_wall_y", size=init_size)
    add_and_name_wall((0, offset, 0), (np.pi/2, 0, 0), "right_wall_y", size=init_size)

    wall_names =  [x[0] + x[1] for x in itertools.product(["left_wall_", "right_wall_"], ["x", "y", "z"])]
    walls = [bpy.data.objects[wall_name]for wall_name in wall_names]

    # Add foreground object
    object_params, obj = load_object(scene_path, args)
    return object_params, obj, walls


def setup_default_scene_geometry(scene_path="", args=None):
    bpy.context.scene.use_nodes = False

    # Get the environment node tree of the current scene
    world_node_tree = bpy.context.scene.world.node_tree
    tree_nodes = world_node_tree.nodes

    # Clear all nodes
    tree_nodes.clear()

    # Add Background node
    node_background = tree_nodes.new(type='ShaderNodeBackground')

    # Add Output node
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = 200,0

    # Link all nodes
    links = world_node_tree.links
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

    # Add background
    bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=(0, 0, -4), scale=(1, 1, 1))
    background = bpy.context.object

    object_params, obj = load_object(scene_path, args)

    return object_params, obj, background

def setup_camera(scene_path, args, background):
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 8), rotation=(0, 0, 0))
    bpy.context.scene.camera = bpy.context.object
    camera = bpy.context.object
    camera.data.sensor_width = 128.0
    camera.data.sensor_height = 128.0    # Square sensor
    if args.focal_length == "large":
        camera.data.lens = 132.2
    elif args.focal_length == "small":
        camera.data.lens = 132.2 * .222
    elif args.focal_length == "medium":
        camera.data.lens = 132.2 * .666
    else:
        util.set_camera_focal_length_in_world_units(camera.data, 525/512*args.resolution) # Set focal length to a common value (kinect)

    background.parent = camera
    background.matrix_parent_inverse = camera.matrix_world.inverted()

    sphere_radius = 1.5
    num_observations = args.num_frames
    bpy.context.scene.frame_end = num_observations

    # TODO: args should define the camera trajectory
    cam_locations = util.sample_controlled_yaw(num_observations, sphere_radius) #get_archimedean_spiral(sphere_radius, num_observations)
    obj_location = np.zeros((1,3))

    cv_poses = util.look_at(cam_locations, obj_location)
    blender_poses = [util.cv_cam2world_to_bcam2world(m) for m in cv_poses]
    for i, pose in enumerate(blender_poses):
        # Write out camera pose
        RT = util.get_world2cam_from_blender_cam(camera)
        cam2world = RT.inverted()

        # insert keyframe for location and rotation
        camera.matrix_world = pose
        camera.keyframe_insert(data_path="location", frame=i+1)
        camera.keyframe_insert(data_path="rotation_euler", frame=i+1)

        pose_dir = os.path.join(scene_path, "poses")
        os.makedirs(pose_dir, exist_ok=True)
        with open(os.path.join(pose_dir, f'{(i+1):06d}.txt'),'w') as pose_file:
            matrix_flat = []
            for j in range(4):
                for k in range(4):
                    matrix_flat.append(cam2world[j][k])
            pose_file.write(' '.join(map(str, matrix_flat)) + '\n')

def delete_all():
    bpy.ops.object.select_all(action="DESELECT")
    objs = bpy.data.objects
    for obj in objs:
        objs.remove(obj, do_unlink=True)

def set_render_settings():
    # Set properties to increase speed of render time
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"  # use cycles for headless rendering
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.cycles.tile_size = 128

    scene.render.image_settings.compression = 0
    scene.cycles.samples = 64
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.max_bounces = 8
    # bpy.context.scene.render.use_persistent_data = True

    scene.cycles.device = "GPU"

    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    device_type = args.device_type
    cycles_preferences.compute_device_type = device_type

    activated_gpus = []
    print(cycles_preferences.get_devices())
    for device in cycles_preferences.devices:
        if not device.type == "CPU":
            print("Activating: ", device)
            device.use = True
            activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = device_type

def setup_midlevel_render_exports(scene_dir):
    # Get the environment node tree of the current scene
    world_node_tree = bpy.context.scene.world.node_tree
    tree_nodes = world_node_tree.nodes

    # Clear all nodes
    tree_nodes.clear()

    # Add Background node
    node_background = tree_nodes.new(type='ShaderNodeBackground')

    # Add Environment Texture node
    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')

    # Load and assign the image to the node property
    cwd = os.getcwd()
    node_environment.image = bpy.data.images.load(cwd + "/multi-area-light.hdr") # Relative path
    node_environment.location = -300,0
    bpy.context.scene.world.cycles.sampling_method = 'MANUAL'
    bpy.context.scene.world.cycles.sample_map_resolution = 128

    # Add Output node
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = 200,0

    # Link all nodes
    links = world_node_tree.links
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1

    objects = bpy.data.objects
    scene = bpy.context.scene

    scene.use_nodes = True

    # Give each object in the scene a unique pass index
    scene.view_layers["ViewLayer"].use_pass_object_index = True
    scene.view_layers["ViewLayer"].use_pass_normal = True
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.view_layers["ViewLayer"].use_pass_mist = True
    scene.view_layers["ViewLayer"].use_pass_vector = True

    for i, obj in enumerate(objects):
        if obj.name == "Plane" or "wall" in obj.name:
            obj.pass_index = 0
        else:
            obj.pass_index = i + 1

    node_tree = scene.node_tree
    links = node_tree.links

    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    # Create a node for the output from the renderer
    compositor_node = node_tree.nodes.new(type="CompositorNodeComposite")
    compositor_node.location = 600, 200

    # Link Image nodes
    render_layers_node = node_tree.nodes.new(type="CompositorNodeRLayers")
    render_layers_node.location = -100, 0
    links.new(render_layers_node.outputs["Image"], compositor_node.inputs["Image"])

    # Mask Output
    mask_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    mask_output_node.label = "Mask_Output"
    mask_output_node.name = "Mask_Output"
    mask_output_node.format.color_mode = "RGB"
    path = os.path.join(scene_dir, "masks")
    mask_output_node.base_path = path
    mask_output_node.location = 600, -200

    # Link Object Index Masks
    links.new(render_layers_node.outputs["IndexOB"], mask_output_node.inputs["Image"])

    # Normals output
    normal_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    normal_output_node.label = "Normal_Output"
    normal_output_node.name = "Normal_Output"
    normal_output_node.format.file_format = "OPEN_EXR"
    normal_output_node.format.color_mode = "RGB"
    path = os.path.join(scene_dir, "normals")
    normal_output_node.base_path = path
    normal_output_node.location = 600, -300

    # Link Normals
    links.new(
        render_layers_node.outputs["Normal"], normal_output_node.inputs["Image"]
    )

    # Depth output
    depth_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    depth_output_node.label = "Depth_Output"
    depth_output_node.name = "Depth_Output"
    depth_output_node.format.file_format = "OPEN_EXR"
    depth_output_node.format.color_mode = "RGB"
    path = os.path.join(scene_dir, "depth")
    depth_output_node.base_path = path
    depth_output_node.location = 600, -100

    normalize_node = node_tree.nodes.new(type="CompositorNodeNormalize")
    normalize_node.location = 400, -100

    # Link Depth
    links.new(render_layers_node.outputs["Depth"], normalize_node.inputs[0])
    links.new(normalize_node.outputs[0], depth_output_node.inputs["Image"])

    # Flow outputs
    flow_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    flow_output_node.label = "Flow_Output"
    flow_output_node.name = "Flow_Output"
    flow_output_node.format.file_format = "OPEN_EXR_MULTILAYER"
    flow_output_node.location = 600, -400
    path = os.path.join(scene_dir, "flows", "Image")
    flow_output_node.base_path = path

    # manually convert to RGBA. See:
    # https://blender.stackexchange.com/questions/175621/incorrect-vector-pass-output-no-alpha-zero-values/175646#175646
    split_rgba = node_tree.nodes.new(type="CompositorNodeSepRGBA")
    combine_rgba = node_tree.nodes.new(type="CompositorNodeCombRGBA")
    split_rgba.location = 200, -400
    combine_rgba.location = 400, -400

    # Link Flows
    links.new(render_layers_node.outputs.get("Vector"), split_rgba.inputs.get("Image"))
    for j, channel in enumerate(["Red", "Green", "Blue", "Alpha"]):
        links.new(split_rgba.outputs[i], combine_rgba.inputs[j])
    links.new(combine_rgba.outputs.get("Image"), flow_output_node.inputs.get("Image"))

    return

def render_scenes(scene_num, args, scene_type):
    scene_path = os.path.join(args.data_dir, f"scene_{scene_num:05d}")
    render_log = os.path.join(scene_path, "render_logs.txt")
    print(os.path.exists(render_log))
    if args.no_overwrite and os.path.exists(scene_path):
        print(f"Render log found for scene: {scene_path}")
        print("Skipping render...")
        return

    bpy.ops.ed.flush_edits()
    delete_all()
    set_render_settings()
    if not os.path.exists(scene_path):
        os.makedirs(scene_path)
    else:
        if os.path.exists(render_log):
            os.remove(render_log)

    texture_configs = sorted(glob("TextureSamples/configs/*.json"))
    if args.textures_per_scene > len(texture_configs):
        print("Not enough textures - create more and try again.")
        sys.exit(1)

    if scene_type == "default":
        obj_params, obj, background = setup_default_scene_geometry(scene_path=scene_path, args=args)
        setup_camera(scene_path, args, background)
    elif scene_type == "ecological":
        obj_params, obj, background_walls = setup_ecological_scene_geometry(scene_path=scene_path, args=args)
        setup_camera(scene_path, args, background_walls[0])

    num_textures_per_scene = args.textures_per_scene
    pose_ids = np.random.choice(range(1, args.max_trajectories + 1), args.num_trajectories, replace=False)
    preset_texture_ids = [0, 7, 13, 16, 21, 22, 25]
    if num_textures_per_scene > len(preset_texture_ids):
        preset_texture_ids = np.random.choice(range(num_textures_per_scene), num_textures_per_scene, replace=False)

    for i in range(num_textures_per_scene):
        texture_id = preset_texture_ids[i]
        texture_config = texture_configs[texture_id]
        texture_name = f"texture_{texture_id:02d}"
        texture_params = json.load(open(texture_config, "r"))
        texture_path = os.path.join(os.getcwd(), scene_path, texture_name)
        if not os.path.exists(texture_path):
            os.makedirs(texture_path)

        foreground_texture = texture_params["foreground"]
        background_texture = texture_params["background"]
        background_scale = 10

        if scene_type == "default":
            background_material = materials.add_material(background_texture, background, "background", seed=scene_num)
            background.scale = (background_scale, background_scale, background_scale)
            texture_params["background"] = background_material
            background.scale = (5, 5, 5) # TODO: Delete this

        elif scene_type == "ecological":
            for k, background in enumerate(background_walls):
                materials.add_material(background_texture, background, f"background_{k}", seed=scene_num)
                background.scale = (background_scale, background_scale, background_scale)

        foreground_obj = bpy.data.objects["Object"]
        foreground_material = materials.add_material(foreground_texture, obj, "foreground", seed=scene_num)
        texture_params["foreground"] = foreground_material

        with open(os.path.join(texture_path, "texture_params.json"), "w") as f:
            json.dump(texture_params, f)

        for p, pose_id in enumerate(pose_ids):
            #pose_dir = sample_and_set_cam_poses(pose_id, num_observations=args.num_frames, sphere_radius=5, all_poses_dir="poses")
            cam_path = os.path.join(texture_path, f"cam_{p:02d}/")
            os.makedirs(cam_path, exist_ok=True)
            # with open(os.path.join(cam_path, "cam_info.txt"), "w") as f:
            #     f.write(f"{pose_dir}")

            bpy.context.scene.render.filepath = cam_path
            if not args.render_pass_only and not args.no_render:
                bpy.ops.render.render(write_still=True, animation=True)

            if args.save_blendfile:
                blend_path = os.path.join(cam_path, f"texture_{i}_cam_{p}.blend")
                bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    # Render shaded + geometry passes
    if scene_type == "default":
        bpy.data.objects.remove(background, do_unlink=True)
    elif scene_type == "ecological":
        for wall in background_walls:
            bpy.data.objects.remove(wall, do_unlink=True)

    bpy.context.scene.render.film_transparent = True

    # Unlink materials
    obj = bpy.data.objects['Object']
    obj.select_set(True)
    obj.data.materials.pop(index=0)

    mask_filepath = os.path.join(scene_path, "render_passes/")
    setup_midlevel_render_exports(mask_filepath)
    bpy.context.scene.render.filepath = os.path.join(scene_path, f"shaded/")

    # Up the quality for ground truth rendering
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.cycles.max_bounces = 8
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoising_prefilter = 'FAST'
    # save object mesh
    # bpy.context.view_layer.objects.active = obj
    blend_path = os.path.join(os.getcwd(), scene_path, "scene.blend")
    #bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    if not args.no_render:
        bpy.ops.render.render(write_still=True, animation=True)

    K = util.get_calibration_matrix_K_from_blender(bpy.data.objects['Camera'].data)
    with open(os.path.join(scene_path, 'intrinsics.txt'),'w') as intrinsics_file:
        intrinsics_file.write('%f %f %f 0.\n'%(K[0][0], K[0][2], K[1][2]))
        intrinsics_file.write('0. 0. 0.\n')
        intrinsics_file.write('1.\n')
        intrinsics_file.write('%d %d\n'%(args.resolution, args.resolution))

    with open(os.path.join(scene_path, "render_log.txt"), "w") as outfile:
        outfile.write(f"Rendering Completed: {datetime.datetime.now()}")

if __name__=="__main__":
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)

    # os.system("nvidia-smi")
    start_scene = args.start_scene
    end_scene = start_scene + args.n_scenes
    scene_type = "default"
    print(f"Rendering scenes: {start_scene}-{end_scene}", flush=True)
    for i in range(start_scene, end_scene):
        render_scenes(i, args, scene_type)
