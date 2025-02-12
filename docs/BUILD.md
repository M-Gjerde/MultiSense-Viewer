## Table of Contents:
- [Code structure](#code-structure)
- [Setting up development environment](#setting-up-development-environment)
    * [Ubuntu](#ubuntu)
    * [Windows](#windows)
        - [Clone this repo using git](#clone-this-repo-using-git)
        - [Tested with Visual Studio 2022.](#tested-with-visual-studio-2022)



# Setting up development environment
## Ubuntu
Install Dependencies:

Open your terminal and type in
```sh
$ sudo apt update
$ sudo apt install build-essential cmake git libzstd-dev libsdl2-dev libgl1-mesa-glx libgl1-mesa-dev libvulkan1 libvulkan-dev libassimp-dev opencl-c-headers libfmt-dev libgtk-3-dev

$ sudo apt install pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libtbb-dev libssl-dev libwebp-dev libsystemd-dev
```
Make sure you have a recent version of the Vulkan SDK installed.  Note that some linux distributions provide this natively.  For example, on Ubuntu 22.04, you can simply run
```sh
$ sudo apt install libvulkan-dev
```
You can also get the vulkan SDK from the Vulkan homepage (https://vulkan.lunarg.com/sdk/home#linux).  For example, here are instructions for installing version 1.3.215 of the Vulkan SDK under Ubuntu 20.04, copied directly from that page:
```sh
$ sudo apt install wget
$ wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
$ sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.216-focal.list https://packages.lunarg.com/vulkan/1.3.216/lunarg-vulkan-1.3.216-focal.list
$ sudo apt update
$ sudo apt install vulkan-sdk
```
Clone the git repo and build the MultiSense Viewer.
```sh
$ git clone --recurse-submodule https://github.com/carnegierobotics/multisense_viewer
$ cd MultiSense-Viewer
$ mkdir build && cd build
$ cmake ..
$ make -j12 # j-parallell jobs (Same number as cores on your cpu for faster compile)
```
Note that for some Ubuntu versions, you may need to allow some warnings in order to achieve a successful build.
```sh
$ git clone --recurse-submodule https://github.com/carnegierobotics/multisense_viewer
$ cmake -DWARNINGS_AS_ERRORS=FALSE ..
$ make -j12
```

In your build folder there is a executable called Multisense-viewer. Launch this to start the application.
To be able to use the auto connect feature, launch with root/admin privileges.

## Windows
<b> Prerequisites: </b>

1. Have Microsoft's Visual studio with c++ desktop environment tools installed. [Link to download](https://visualstudio.microsoft.com/vs/) <br/>
2. Install the Vulkan SDK for windows. Get the SDK installer [here](https://sdk.lunarg.com/sdk/download/1.3.216.0/windows/VulkanSDK-1.3.216.0-Installer.exe) and run it. <br/>
3. Install WinPcap DLLs. [link to installer](https://www.winpcap.org/install/bin/WinPcap_4_1_3.exe). You can uncheck enabling winpcap driver after installation.
4. Install OpenSSL

### Clone this repo and submodules using git
``` sh
$ git clone --recurse-submodule https://github.com/carnegierobotics/multisense_viewer
```
Then choose to compile through Visual Studio or CMAKE Command Line Interface

#### Compile Visual Studio 2022.
1. Open up a new cmake project in Visual studio. Remember to launch as admin.
2. Configure the cmake file and set MultiSense-viewer.exe as startup item
3. Launch using VS or run the exe located in out/ folder

#### Compile Using CMAKE
```sh
$ cd MultiSense-Viewer
$ mkdir build && cd build
$ cmake -DWARNINGS_AS_ERRORS=FALSE ..
$ cmake --build .
```
