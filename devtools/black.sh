#!/bin/bash

# Check if the user provided a directory argument
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Format the specified directory with Black
black "$1" --line-length 88