#!/bin/bash
#SBATCH --job-name texture-scene-gen
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=24GB
#SBATCH -p tenenbaum
#SBATCH --mem=2G
#SBATCH --array=0-10%5
#SBATCH --output=/om2/user/yyf/%x.%A_%a.log

IDX=$SLURM_ARRAY_TASK_ID
START_SCENE=$((IDX * 1000))
/om/user/yyf/blender-3.2.1-linux-x64/blender -b -noaudio -P generate_scene.py -- --start_scene $START_SCENE --n_scenes 1000 --data_dir /om2/user/yyf/textured-shapes/scenes/
