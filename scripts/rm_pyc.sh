#!/bin/bash

# Check if a directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <starting_directory>"
    exit 1
fi

# Get the starting directory from the first argument
START_DIR="$1"

# Check if the provided argument is a valid directory
if [ ! -d "$START_DIR" ]; then
    echo "Error: '$START_DIR' is not a valid directory."
    exit 1
fi

# Find and remove all __pycache__ directories and their contents
echo "Removing all __pycache__ directories from $START_DIR ..."
find "$START_DIR" -type d -name "__pycache__" -exec rm -rf {} +

echo "All __pycache__ directories have been removed."
