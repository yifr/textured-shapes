import bpy
import os
import glob

def load_scene(scene_dir):
    filepath = os.path.join(scene_dir, 'scene.blend')
    bpy.ops.wm.open_mainfile(filepath=filepath)

def export_obj(scene_dir):
    load_scene(scene_dir)
    obj_name = "Object"
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[obj_name].select_set(True)

    bpy.ops.export_scene.obj(filepath=os.path.join(scene_dir, 'scene.obj'), use_selection=True)

def main():
    scenes = glob.glob("/om2/user/yyf/textured-shapes/hi-res-scenes/*")
    for scene_dir in scenes:
        print(scene_dir)
        export_obj(scene_dir, "/Users/yonifriedman/Projects/textured-shapes/scenes/scene_00000/")

main()