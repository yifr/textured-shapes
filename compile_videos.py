import os
import sys
import subprocess
from glob import glob
from PIL import Image
import argparse
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="scenes", help="root directory with all the scenes")
parser.add_argument("--output_dir", type=str, default="movies")
parser.add_argument("--start_scene", type=int, default=0)
parser.add_argument("--num_scenes", type=int, default=-1, help="If greater than 0, will only render num_scenes scenes")
parser.add_argument("--experiment_name", type=str, default="", help="Will append this string to the start of the output file")
parser.add_argument("--frame_rate", type=int, default=15)
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


def create_video(input_images, output_path, frame_rate):
    images = []
    for img_path in input_images:
        images.append(imageio.imread(img_path))
    print("Saving gif to: ", output_path)
    imageio.mimsave(output_path, images, fps=frame_rate)


scene_dir = os.path.join(args.root_dir, "scene_*")
scenes = sorted(glob(scene_dir))
scenes = scenes[args.start_scene:]
print(f"Found {len(scenes)} scenes at path: {scene_dir}")
for i, scene_path in enumerate(scenes):
    if args.num_scenes > 0:
        if i == args.num_scenes:
            print(f"Finished stitching videos on {args.num_scenes} scenes")
            break

    # Render shaded video
    img_paths = sorted(glob(os.path.join(scene_path, "shaded", "*.png")))
    movie_name = scene_path.split("/")[-1] + "-shaded"
    if args.experiment_name:
        movie_name = args.experiment_name + "-" + movie_name
    output_path = os.path.join(args.output_dir, f"{movie_name}.gif")
    create_video(img_paths, output_path, args.frame_rate)

    # Render textured videos
    texture_dirs = glob(os.path.join(scene_path, "texture*"))
    for texture_dir in texture_dirs:
        movie_names = texture_dir.split("/") # should be something like: root_dir/scene_num/texture_dir
        movie_name = "-".join(movie_names[-2:])
        if args.experiment_name:
            movie_name = args.experiment_name + "-" + movie_name
        output_path = os.path.join(args.output_dir, f"{movie_name}.gif")
        img_paths = sorted(glob(os.path.join(texture_dir, "*.png")))
        create_video(img_paths, output_path, args.frame_rate)

        # Check for FlowFormer videos
        texture_flows = glob(os.path.join(texture_dir, "FlowFormer", "*.png"))
        if len(texture_flows) > 0:
            img_paths = sorted(glob(os.path.join(texture_dir, "FlowFormer", "*.png")))
            output_path = os.path.join(args.output_dir, f"{movie_name}-flow.gif")
            create_video(img_paths, output_path, args.frame_rate)

