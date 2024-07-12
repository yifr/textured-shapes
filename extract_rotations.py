import os
import json
import bpy

# Set the base directory containing the scene folders
base_dir = "/om2/user/yyf/textured-shapes/hi-res-scenes/"

# Get a list of all scene directories
scene_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('scene_')]

# Function to get the rotation_euler from the shape.obj in Blender
def get_rotation_euler(scene_blend_path):
    bpy.ops.wm.open_mainfile(filepath=scene_blend_path)
    obj = bpy.data.objects.get('Object')  # Assuming the object name is 'shape'
    if obj is not None:
        rotation_euler = obj.rotation_euler
        return [rotation_euler.x, rotation_euler.y, rotation_euler.z]
    else:
        raise ValueError("No object named 'object' found in the blend file.")

# Iterate through each scene directory and update the json file
for scene_dir in scene_dirs:
    scene_path = os.path.join(base_dir, scene_dir)
    scene_blend_path = os.path.join(scene_path, 'scene.blend')
    json_file_path = os.path.join(scene_path, 'obj_params.json')
    updated_json_file_path = os.path.join(scene_path, 'obj_params_updated.json')
    try:
        rotation_euler = get_rotation_euler(scene_blend_path)

        # Read the existing obj_params.json file
        with open(json_file_path, 'r') as json_file:
            obj_params = json.load(json_file)

        # Add the rotation_euler to the obj_params
        obj_params['rotation_euler'] = rotation_euler

        # Write the updated params back to the json file
        with open(updated_json_file_path, 'w') as json_file:
            json.dump(obj_params, json_file)

        print(f"Updated {updated_json_file_path} with rotation_euler: {rotation_euler}")

    except Exception as e:
        print(f"Error processing {scene_dir}: {e}")

print("Finished updating all scenes.")
