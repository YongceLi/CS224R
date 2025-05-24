#!/bin/bash

# Exit on error
set -e

echo "Setting up dependencies for CS224R..."

# Add NVIDIA repository and update
echo "Adding NVIDIA repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CuDNN for CUDA 12
echo "Installing CuDNN..."
sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# Install OpenGL and OSMesa dependencies
echo "Installing OpenGL and OSMesa dependencies..."
sudo apt-get install -y libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev

# Install Python OpenGL packages
echo "Installing Python OpenGL packages..."
pip install PyOpenGL PyOpenGL-accelerate

echo "Setup completed successfully!" 