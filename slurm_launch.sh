#!/bin/bash
#SBATCH --job-name scenegen
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 48:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -p tenenbaum
#SBATCH --mem=2G
#SBATCH --array=0-20
#SBATCH --output=/om2/user/yyf/%x.%A_%a.log

IDX=$SLURM_ARRAY_TASK_ID
START_SCENE=$((IDX * 250))
/scratch2/weka/tenenbaum/yyf/blender-3.2.1-linux-x64/blender -b -noaudio -P generate_scene.py -- --start_scene $START_SCENE --n_scenes 250 --data_dir /om2/user/yyf/textured-shapes/training-set/ --resolution 512 --shape_type shapegen --max_trajectories 1 --num_trajectories 1 --textures_per_scene 5
