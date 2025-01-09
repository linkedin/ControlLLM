# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

#!/bin/bash

# Check if an argument is provided and extract EVAL_PATH if so
if [ $# -eq 1 ]; then
    EVAL_PATH="${1#*=}"
else
    # Prompt the user for the EVAL_PATH if not provided as an argument
    read -p "Enter the absolute path to the lm-evaluation-harness: " EVAL_PATH
fi

conda activate 

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Use the script's directory to find the open_llm_leaderboard directory
DIR="$SCRIPT_DIR/open_llm_leaderboard"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' not found."
    exit 1
fi

# Iterate over YAML files in the directory and update them
for YAML_FILE in "$DIR"/*.yaml
do
    if [ -f "$YAML_FILE" ]; then
        sed -i 's|{\$EVAL_PATH}|'"$EVAL_PATH"'|g' "$YAML_FILE"
        echo "Updated $YAML_FILE with EVAL_PATH: $EVAL_PATH"
    fi
done
