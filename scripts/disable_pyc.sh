#!/bin/bash

# Detect the user's shell (bash or zsh)
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
else
    echo "Unsupported shell. This script only supports bash or zsh."
    exit 1
fi

# Define the line to be added to the shell configuration file
LINE="export PYTHONDONTWRITEBYTECODE=1"

# Check if the line is already present
if grep -Fxq "$LINE" "$SHELL_RC"; then
    echo "PYTHONDONTWRITEBYTECODE is already set in $SHELL_RC."
else
    # Append the line to the shell configuration file
    echo "$LINE" >> "$SHELL_RC"
    echo "Added PYTHONDONTWRITEBYTECODE=1 to $SHELL_RC."
fi

# Reload the shell configuration
echo "Reloading shell configuration..."
source "$SHELL_RC"

echo "PYTHONDONTWRITEBYTECODE is now set permanently for your shell."
