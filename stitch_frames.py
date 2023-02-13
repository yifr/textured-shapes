import os
import sys
import subprocess
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="scenes", help="root directory with all the scenes")
parser.add_argument("--output_dir", type=str, default="movies")
parser.add_argument("--start_scene", type=int, default=0)
parser.add_argument("--num_scenes", type=int, default=-1, help="If greater than 0, will only render num_scenes scenes")
parser.add_argument("--experiment_name", type=str, default="", help="Will append this string to the start of the output file")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

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
    shaded_imgs = os.path.join(scene_path, "shaded", "%04d.png")
    movie_name = scene_path.split("/")[-1] + "-shaded.mp4"
    if args.experiment_name:
        movie_name = args.experiment_name + "-" + movie_name

    output_path = os.path.join(args.output_dir, f"{movie_name}.mp4")
    try:
        subprocess.run([
            f"ffmpeg","-y", "-framerate", "16", "-i",
            f"{shaded_imgs}", "-pix_fmt", "yuv420p",  "-c:v", "libx264", f"{output_path}"], check=True
        )
    except Exception as e:
        print(e)
        sys.exit(1)

    # Render textured videos
    texture_dirs = glob(os.path.join(scene_path, "texture*"))
    for texture_dir in texture_dirs:
        movie_names = texture_dir.split("/") # should be something like: root_dir/scene_num/texture_dir
        movie_name = "-".join(movie_names[-2:])
        if args.experiment_name:
            movie_name = args.experiment_name + "-" + movie_name
        output_path = os.path.join(args.output_dir, f"{movie_name}.mp4")
        texture_imgs = os.path.join(texture_dir, "%04d.png")
        try:
            subprocess.run([
                f"ffmpeg","-y", "-framerate", "16", "-i",
                f"{texture_imgs}", "-pix_fmt", "yuv420p",  "-c:v", "libx264", f"{output_path}"], check=True
            )
        except Exception as e:
            print(e)
            sys.exit(1)

        # Check for FlowFormer videos
        texture_flows = glob(os.path.join(texture_dir, "FlowFormer", "*.png"))
        if len(texture_flows) > 0:
            flow_imgs = os.path.join(texture_dir, "FlowFormer", "%04d.png")
            output_path = os.path.join(args.output_dir, f"{movie_name}-flow.mp4")
            try:
                subprocess.run([
                    f"ffmpeg","-y", "-framerate", "16", "-i",
                    f"{flow_imgs}", "-pix_fmt", "yuv420p",  "-c:v", "libx264", f"{output_path}"], check=True
                )
            except Exception as e:
                print(e)
                sys.exit(1)

