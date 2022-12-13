import os
import bpy
import json
from glob import glob

"""
Blender script to record the texture parameters of a foreground object
and a background plane
"""
TEXTURES = ["Wave Texture", "Noise Texture", "Voronoi Texture", "Magic Texture", "Musgrave Texture"]

# Specify relevant fields to record for different textures:
obj_name = 'Cube'
obj = bpy.data.objects[obj_name]
background = bpy.data.objects['Plane']

# Get the texture parameters
obj_tex = obj.active_material.node_tree.nodes
background_tex = background.active_material.node_tree.nodes

def get_output_name(texture):
    for output in texture.outputs:
        output_name = output.name
        if texture.outputs[output_name].is_linked:
            print(output_name, "Linked")
            return output_name
    print(texture.name, "No linked output")
    return None

def get_texture_params(node_tree, texture_type):
    texture_params = {}
    texture = node_tree[texture_type]
    output_name = get_output_name(texture)
    input_node = texture.inputs["Vector"].links[0].from_node.name
    output_node = texture.outputs[output_name].links[0].to_node.name
   
    inputs = {}
    for input in texture.inputs:
        if input.name == "Vector":
            continue
        inputs[input.name] = texture.inputs[input.name].default_value

    texture_params["input_values"] = inputs
    texture_params["texture_type"] = texture_type
    texture_params["output_name"] = output_name     # Name of the output feature 
    texture_params["input_node"] = input_node       # Node the texture gets input from
    texture_params["output_node"] = output_node     # Node that the texture is connected to
    texture_params["input_values"] = inputs         # Input values for the texture (scale, distortion, etc...)

    if texture_type == "Voronoi Texture":
        feature = texture.feature
        distance = texture.distance
        texture_params["feature"] = feature             
        texture_params["distance"] = distance           

    elif texture_type == "Musgrave Texture":
        musgrave_type = texture.musgrave_type
        texture_params["musgrave_type"] = musgrave_type

    elif texture_type == "Magic Texture":
        turbulence_depth = texture.turbulence_depth
        texture_params["turbulence_depth"] = turbulence_depth

    elif texture_type == "Wave Texture":
        wave_type = texture.wave_type
        wave_profile = texture.wave_profile
        rings_direction = texture.rings_direction
        
        texture_params["wave_type"] = wave_type
        texture_params["wave_profile"] = wave_profile
        texture_params["rings_direction"] = rings_direction

    return texture_params


scene_textures = {"foreground": {}, "background": {}}
for texture in TEXTURES:
    if texture in obj_tex:
        print(texture)
        texture_params = get_texture_params(obj_tex, texture)
        scene_textures["foreground"][texture] = texture_params

    if texture in background_tex:
        texture_params = get_texture_params(background_tex, texture)
        scene_textures["background"][texture] = texture_params

print(scene_textures)

config_path = "/Users/yonifriedman/Projects/GestaltNeurophys/TextureSamples/configs/"
existing_texures = glob(config_path + "*")
num_textures = len(existing_texures)
with open(os.path.join(config_path, f"texture_{num_textures:03d}.json"), "w") as f:
    json.dump(existing_texures, f)