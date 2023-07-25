# Run.sh - takes from a config directory and runs the pl_trainevaltestsave.py script on each config file in the directory
# This is only useful if you want to run a bunch of experiments at once
# Ryan Peruski, 07/21/23

#!/bin/bash
# conda activate pytorch
config_directory="../config"

# Iterate over each config file in the directory
for config_file in "$config_directory"/*.py; do
    if [ "$(basename "$config_file")" = "empty.py" ]; then
        continue
    fi
    echo "Running with config file: $config_file"
    python src/pl_trainevaltestsave.py $config_file
    echo "-------------------------------------"
done