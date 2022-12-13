import os
import sys
import subprocess
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--glob_dir", type=str, default="scenes")
parser.add_argument("--output_dir", type=str, default="movies")
args = parser.parse_args()

scene_dir = os.path.join(args.glob_dir, "*/*/")
print(scene_dir)
scenes = glob(scene_dir)
print("Found ", len(scenes), " scenes")
for i, scene_path in enumerate(scenes):
    movie_names = scene_path.split("/")
    movie_name = "_".join(movie_names[-3:-1])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(scene_path)
    output_path = os.path.join(args.output_dir, f"{movie_name}.mp4")
    try:
        subprocess.run([
            f"ffmpeg","-y", "-framerate", "16", "-i",
            f"{scene_path}/%04d.png", "-pix_fmt", "yuv420p",  "-c:v", "libx264", f"{output_path}"], check=True
        )
    except Exception as e:
        print(e)
        sys.exit(1)
