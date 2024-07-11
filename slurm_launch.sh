#!/bin/bash
#SBATCH --job-name scenegen
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p tenenbaum
#SBATCH --mem=2G
#SBATCH --array=0-1
#SBATCH --output=/om2/user/yyf/logs/%x.%A_%a.log

IDX=$SLURM_ARRAY_TASK_ID
START_SCENE=$((IDX * 20))
#/scratch2/weka/tenenbaum/yyf/blender-3.2.1-linux-x64/blender -b -noaudio -P generate_scene.py -- --start_scene $START_SCENE --n_scenes 20 --data_dir /om2/user/yyf/textured-shapes/texture-sweep/ --resolution 512 --shape_type shapegen --max_trajectories 1 --num_trajectories 1 --textures_per_scene 28 --no_overwrite

if [$IDX -eq 0]; then

	# small zoom
	/scratch2/weka/tenenbaum/yyf/blender-3.2.1-linux-x64/blender -b -noaudio -P generate_scene.py -- --start_scene 0 --n_scenes 20 --data_dir /om2/user/yyf/textured-shapes/scaled-hi-res/small/ --resolution 1024 --shape_type shapegen --max_trajectories 1 --num_trajectories 1 --textures_per_scene 7 --alternate_obj_directory /om2/user/yyf/textured-shapes/hi-res-scenes/ --focal_length small

else
	# medium zoom
	/scratch2/weka/tenenbaum/yyf/blender-3.2.1-linux-x64/blender -b -noaudio -P generate_scene.py -- --start_scene 0 --n_scenes 20 --data_dir /om2/user/yyf/textured-shapes/scaled-hi-res/medium/ --resolution 1024 --shape_type shapegen --max_trajectories 1 --num_trajectories 1 --textures_per_scene 7 --alternate_obj_directory /om2/user/yyf/textured-shapes/hi-res-scenes/ --focal_length medium
fi
