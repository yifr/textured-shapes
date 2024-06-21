#!/bin/bash

root_dir=/om2/user/yyf/textured-shapes/hi-res-scenes/
frame_rate=60


# Find all scene directories
scene_directories=$(find $root_dir -type d -name "scene_*")

# Loop through each scene directory
for scene_dir in $scene_directories; do
  # Find "unshaded" and "texture_" directories in the current scene directory
  texture_directories=$(find "$scene_dir" -mindepth 1 -type d \( -name "shaded" \)) #-o -name "texture_*" \))
  scene_name=$(basename "$scene_dir")
  # Loop through each texture directory
  i=0
  for texture_dir in $texture_directories; do
    texture_name=$(basename "$texture_dir")

    # Create a temporary text file listing all the input images
    printf "file '%s'\n" $texture_dir/*.png > input_list.txt

    # Use FFmpeg to stitch together the images into a video
    ffmpeg -y -r $frame_rate -f image2 -pattern_type glob -i "${texture_dir}/*.png" -c:v libx264 -pix_fmt yuv420p -crf 23 -r $frame_rate "/om2/user/yyf/textured-shapes/output_vids/${scene_name}_${texture_name}_video.mp4"

    # Remove the temporary text file
    rm input_list.txt
    $i = $(($i + 1))
    if $i > 10; then
        return
    fi
  done
done

