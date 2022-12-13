#!/bin/bash
#SBATCH --job-name texture-scene-gen
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yyf@mit.edu
#SBATCH -t 94:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=24GB
#SBATCH -p tenenbaum
#SBATCH --mem=4G
#SBATCH --output=/om2/user/yyf/%x.%A_%a.log

/om/user/yyf/blender-3.2.1-linux-x64/blender -b -noaudio -P generate_scene.py
