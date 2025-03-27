#!/bin/bash
# CUDA 12.4 Installation Script
# This script installs CUDA 12.4 on Ubuntu systems
# Run with sudo privileges: sudo bash cuda_install.sh

set -e  # Exit on error
set -u  # Treat unset variables as errors

# Print colorful messages
print_message() {
    echo -e "\e[1;34m[CUDA Installer] $1\e[0m"
}

print_error() {
    echo -e "\e[1;31m[ERROR] $1\e[0m"
}

print_success() {
    echo -e "\e[1;32m[SUCCESS] $1\e[0m"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run this script with sudo privileges"
    exit 1
fi

# Check for Ubuntu
if [ ! -f /etc/lsb-release ]; then
    print_error "This script is intended for Ubuntu systems"
    print_message "For other distributions, please check NVIDIA documentation"
    exit 1
fi

# Check system architecture
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    print_error "This script supports only x86_64 architecture"
    exit 1
fi

print_message "Starting CUDA 12.4 installation..."

# Update system
print_message "Updating system packages..."
apt-get update
apt-get upgrade -y

# Install required packages
print_message "Installing required dependencies..."
apt-get install -y build-essential dkms
apt-get install -y linux-headers-$(uname -r)

# Remove old NVIDIA drivers if present
print_message "Checking for existing NVIDIA installations..."
apt-get remove --purge -y nvidia* cuda* > /dev/null 2>&1 || true
apt autoremove -y

# Add NVIDIA repository
print_message "Adding NVIDIA repository..."
apt-get install -y software-properties-common
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
apt-get update

# Install CUDA 12.4
print_message "Installing CUDA 12.4. This might take some time..."
apt-get install -y cuda-12-4

# Set up environment variables
print_message "Setting up environment variables..."
cat > /etc/profile.d/cuda.sh << 'EOF'
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF

# Make environment variables available immediately
source /etc/profile.d/cuda.sh

# Verify installation
print_message "Verifying CUDA installation..."
if [ -f /usr/local/cuda-12.4/bin/nvcc ]; then
    NVCC_VERSION=$(/usr/local/cuda-12.4/bin/nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    if [[ "$NVCC_VERSION" == 12.4* ]]; then
        print_success "CUDA 12.4 installation completed successfully!"
        print_message "CUDA location: /usr/local/cuda-12.4/"
        print_message "NVCC version: $NVCC_VERSION"
    else
        print_error "CUDA installation version mismatch: $NVCC_VERSION"
    fi
else
    print_error "CUDA installation failed. NVCC not found."
    exit 1
fi

# Instructions for the user
print_message "To use CUDA in your current session, run:"
echo "    source /etc/profile.d/cuda.sh"

print_message "To verify the installation, run:"
echo "    nvcc --version"

print_message "To test the CUDA installation with a sample, run:"
echo "    cd /usr/local/cuda-12.4/samples/1_Utilities/deviceQuery"
echo "    make"
echo "    ./deviceQuery"

print_success "CUDA 12.4 setup completed!"