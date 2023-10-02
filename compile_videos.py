import os
import sys
import subprocess
from glob import glob
from PIL import Image
import argparse
from collections import deque
import imageio.v2 as imageio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="scenes", help="root directory with all the scenes")
parser.add_argument("--output_dir", type=str, default="movies")
parser.add_argument("--start_scene", type=int, default=0)
parser.add_argument("--num_scenes", type=int, default=-1, help="If greater than 0, will only render num_scenes scenes")
parser.add_argument("--experiment_name", type=str, default="", help="Will append this string to the start of the output file")
parser.add_argument("--frame_rate", type=int, default=10)
parser.add_argument("--no_flow", action="store_true")

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    

def create_video(input_images, output_path, frame_rate):
    images = []
    writer = imageio.get_writer(output_path, format='FFMPEG', mode='I', fps=frame_rate)
    for img_path in input_images:
        writer.append_data(imageio.imread(img_path))
    writer.close()
    return


# Conditions:
# 1a. N Frames: [2, full]
# 1b. Aligned across texture
#2. Consecutive frames


def shift(frame_list, shift_index=-1, min_shift=10):
    """
    Shifts a list of frames by a pre-set (or randomly selected) number of frames.
    Currently shifts the entire
    """
    n_frames = len(frame_list)
    if shift_index == -1:
        if min_shift < n_frames:
            print("min_shift parameter must be less than the number of frames in the video")
            sys.exit(1)

        shift_index = np.random.randint(min_shift, n_frames - 1)

    new_frame_list = []
    for i in range(len(frame_list)):
        if shift_index == n_frames:
            shift_index = 0
        new_frame_list.append(frame_list[shift_index])
        shift_index += 1

    return new_frame_list, shift_index + 1

def compile_all_videos(scene_path, args):
    n_frames_list = [90]
    motion_alignment = [True, False]
    frame_jumps = [0]
    shift_frames = [5, 8, 11, 14, 17, 20, 23]

    # Render shaded video
    img_paths = sorted(glob(os.path.join(scene_path, "shaded", "*.png")))
    movie_name = scene_path.split("/")[-1] + "-shaded"
    if args.experiment_name:
        movie_name = args.experiment_name + "-" + movie_name

    for n_frames in n_frames_list:
        for alignment in motion_alignment:

            if not os.path.exists(directory_path):
                os.makedirs(directory_path, exist_ok=True)

            if alignment:
                frames = img_paths
            else:
                frames, shift_idx = shift(frames, min_shift=5)

            directory_path = os.path.join(args.output_dir, f"aligned={alignment}_startframe={shift_idx}")
            output_path = os.path.join(directory_path, f"{movie_name}.gif")
            frames = get_frames(img_paths, n_frames, alignment, skip_frames, 0)
            create_video(frames, output_path, np.min((len(frames), args.frame_rate)))


    # Render textured videos
    texture_dirs = sorted(glob(os.path.join(scene_path, "texture*")))
    for i, texture_dir in enumerate(texture_dirs):
        shifts = shift_frames[i]
        movie_names = texture_dir.split("/") # should be something like: root_dir/scene_num/texture_dir
        movie_name = "-".join(movie_names[-2:])
        if args.experiment_name:
            movie_name = args.experiment_name + "-" + movie_name

        output_path = os.path.join(args.output_dir, f"{movie_name}.gif")
        img_paths = sorted(glob(os.path.join(texture_dir, "*.png")))
        for n_frames in n_frames_list:
            for alignment in motion_alignment:
                for skip_frames in frame_jumps:
                    directory_path = os.path.join(args.output_dir, f"frames={n_frames}_aligned={alignment}_skipframes={skip_frames}")
                    output_path = os.path.join(directory_path, f"{movie_name}.gif")
                    frames = get_frames(img_paths, n_frames, alignment, skip_frames, shifts)
                    create_video(frames, output_path, np.min((len(frames), args.frame_rate)))


        # Check for FlowFormer videos
        if args.no_flow:
            return

        img_paths = sorted(glob(os.path.join(texture_dir, "FlowFormer", "*.png")))
        if len(img_paths) > 0:
            for n_frames in n_frames_list:
                for alignment in motion_alignment:
                    for skip_frames in frame_jumps:
                        directory_path = os.path.join(args.output_dir, f"frames={n_frames}_aligned={alignment}_skipframes={skipframes}")
                        output_path = os.path.join(directory_path, f"{movie_name}-flow.gif")
                        frames = get_frames(img_paths, n_frames, alignment, skip_frames, shift_frames)

                        create_video(frames, output_path, np.min((len(frames), args.frame_rate)))

    return

def render_videos(scene_path, args):
    img_paths = sorted(glob(os.path.join(scene_path, "shaded", "*.png")))
    movie_name = scene_path.split("/")[-1] + "-shaded"
    output_path = os.path.join(args.output_dir, f"{movie_name}.mp4")
    print(output_path)
    create_video(img_paths, output_path, 15)

    texture_dirs = sorted(glob(os.path.join(scene_path, "texture*")))
    for texture_dir in texture_dirs:
        img_paths = sorted(glob(os.path.join(texture_dir, "*.png")))
        print(texture_dir)
        movie_name = "-".join(texture_dir.split("/")[-2:]) 
        output_path = os.path.join(args.output_dir, f"{movie_name}.mp4")
        print(output_path)
        create_video(img_paths, output_path, 15)


if __name__=="__main__":
    scene_dir = os.path.join(args.root_dir, "scene_*")
    scenes = sorted(glob(scene_dir))
    scenes = scenes[args.start_scene:]
    print(f"Found {len(scenes)} scenes at path: {scene_dir}")
    for i, scene_path in enumerate(scenes):
        if args.num_scenes > 0:
            if i == args.num_scenes:
                print(f"Finished stitching videos on {args.num_scenes} scenes")
                break
            render_videos(scene_path, args)
            #compile_all_videos(scene_path, args)

