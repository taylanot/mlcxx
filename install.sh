#!/bin/bash

# Exit on any error
set -e

# Define source files/directories
HEADER_FILE="lcpp.h"
HEADER_DIR="lcpp_bits"

# Define target include directory
TARGET_DIR="/usr/local/include"

# Check if running as root (required to write to /usr/local/include)
if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root. Try: sudo ./install.sh"
  exit 1
fi

# Copy the main header
echo "Copying $HEADER_FILE to $TARGET_DIR..."
cp -v "$HEADER_FILE" "$TARGET_DIR/"

# Copy the header bits directory
echo "Copying $HEADER_DIR to $TARGET_DIR..."
cp -rv "$HEADER_DIR" "$TARGET_DIR/"

echo "Installation complete! You can now include <lcpp.h> in your projects."
