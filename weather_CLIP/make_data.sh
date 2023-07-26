#!/bin/bash

# Replace "/path/to/input/directory" with the actual path of your input directory
input_directory="/root/BlurImageTrainingProject/weather_videos_for_ornl"

# Replace "/path/to/output/directory" with the actual path of your output directory
output_directory="/root/BlurImageTrainingProject/new_weather_pictures_for_ornl"

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# Loop through each file in the input directory (recursively)
find "$input_directory" -type f -print0 | while IFS= read -r -d '' input_file; do
    # Get the relative path of the input file
    relative_path="${input_file#$input_directory}"

    # Remove leading slash if present (necessary for concatenation)
    relative_path="${relative_path#/}"

    # Generate the output file path by concatenating output_directory and relative_path
    output_file="$output_directory/$relative_path"

    # Create the output directory for the current file if it doesn't exist
    mkdir -p "$(dirname "$output_file")"

    # Run ffmpeg on the current file
    ffmpeg -i "$input_file" "$output_file%7d.png" > /dev/null 2>&1

    echo "Processed: $input_file"
done

echo "All files processed!"