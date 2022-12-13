import numpy as np
from PIL import Image, ImageDraw

try:
    import bpy
except ImportError:
    print(
        "Unable to import Blender Python Interface (bpy). \
        No procedural textures will be available."
    )


PROCEDURAL_TEXTURES = ["Voronoi Texture", "Wave Texture", "Musgrave Texture", "Noise Texture", "Magic Texture"]

def default_material(obj, color=None, material_name="default"):
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]

    if color == "random":
        color = np.random.rand(4)
        color[-1] = 1

    bsdf.inputs[0].default_value = color

    nodes["Material Output"].location = (800, 0)
    bsdf.location = (600, 0)

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return


def Shader(name):
    name = name.split(" ")[0]
    return "ShaderNodeTex" + name


def add_material(texture_config, obj=None, material_name="Material"):
    """
    Adds texture config to an object
    """
    scene = bpy.context.scene
    if obj == None:
        obj = scene.context.view_layer.objects.active

    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    nodes.remove(bsdf)

    emission_node = nodes.new(type="ShaderNodeEmission")

    texture_nodes = [node for node in texture_config.keys() if "Texture" in node]
    textures = {}

    color_ramp = nodes.new(type="ShaderNodeValToRGB")
    mapping_node = nodes.new(type="ShaderNodeMapping")
    coordinate_node = nodes.new(type="ShaderNodeTexCoord")

    for i, texture_name in enumerate(texture_nodes):
        config = texture_config[texture_name]
        texture_type = Shader(texture_name)
        texture = nodes.new(type=texture_type)
        texture.location = (i * 200, 0)
        textures[texture_name] = texture

        if texture_name == "Voronoi Texture":
            texture.distance = config["distance"]
            texture.feature = config["feature"]

        elif texture_name == "Musgrave Texture":
            texture.musgrave_type = config["musgrave_type"]

        elif texture_name == "Magic Texture":
            texture.turbulence_depth = config["Turblence Depth"]

        elif texture_name == "Wave Texture":
            texture.wave_profile = config["wave_profile"]
            texture.wave_type = config["wave_type"]
            texture.rings_direction = config["rings_direction"]

        for input_val in config["input_values"]:
            texture.inputs[input_val].default_value = config["input_values"][input_val]

    # Add random noise to texture generator
    mapping_node.inputs["Location"].default_value = np.random.randint(-1000, 1000, 3)
    mapping_node.inputs["Rotation"].default_value = np.random.randint(0, 360, 3)

    nodes["Material Output"].location = (800, 0)
    emission_node.location = (600, 0)
    color_ramp.location = (400, 0)

    mapping_node.location = (-200, 0)
    coordinate_node.location = (-400, 0)

    ##############
    # Link Nodes
    ##############
    links = mat.node_tree.links
    links.new(emission_node.outputs[0], nodes["Material Output"].inputs[0])

    # Coordinate Texture -> Mapping Node
    links.new(coordinate_node.outputs["Object"], mapping_node.inputs[0])

    # Link texture nodes:
    for i, texture_name in enumerate(texture_nodes):
        config = texture_config[texture_name]
        upstream_node = config["input_node"]
        downstream_node = config["output_node"]
        output_feature = config["output_name"]
        texture = textures[texture_name]

        if upstream_node == "Mapping":
            links.new(mapping_node.outputs[0], texture.inputs[0])

        if downstream_node == "ColorRamp":
            links.new(texture.outputs[output_feature], color_ramp.inputs[0])
        else:
            downstream_node = textures[downstream_node]
            links.new(texture.outputs[output_feature], downstream_node.inputs[0])

    links.new(color_ramp.outputs["Color"], emission_node.inputs[0])
    color_ramp.color_ramp.elements.new(0.5)

    # Currently only support black and white colors
    color_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    color_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)

    # Evenly interpolate between color/white spots
    color_ramp.color_ramp.elements[0].position = 0
    color_ramp.color_ramp.elements[1].position = 0.5
    color_ramp.color_ramp.interpolation = "CONSTANT"

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return
