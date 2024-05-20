# Installation

To run the program, you need:
- an Nvidia GPU compatible with CUDA
- Nvidia drivers installed
- CUDA installed

## Install Nvidia drivers

### Windows

https://www.nvidia.com/download/index.aspx

### Ubuntu

Get the list of available drivers:  
``sudo ubuntu-drivers list``

Then either let the program install the driver that it thinks is the best fitted:  
``sudo ubuntu-drivers install``

Or choose your own version that you want to install from the list previously talked about:  
``sudo ubuntu-drivers install nvidia:{VERSION}``

You can check that the driver was correctly installed using this command:  
``nvidia-smi``

It will display information about GPU usage, as well as the Driver version that you installed.  
If the command doesn't work after you installed the drivers, try to restart your computer.

### Other Linux distributions

Normally, the CUDA toolkit installation (detailed later) will also be able to install an nvidia driver automatically. However, you won't be able to choose the version, and in my case, it wasn't compatible with my GPU.

## Install CUDA toolkit

### Windows

Follow instructions:  
https://developer.nvidia.com/cuda-downloads

### Linux (all distributions)

Verify that your distribution supports CUDA:  
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

If it does, follow instructions on this website:  
https://developer.nvidia.com/cuda-downloads

## My configuration

Just in case, here is my configuration:

GTX 1080Ti  
Driver version: 535.171.04  
CUDA version: 12.4  
CMake version: 3.29.3
