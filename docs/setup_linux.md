# Setup (Linux)

The following instructions describe how to install Vulkan Glasses and COLMAP, and how to set up the `semantic_ba_nature_alain` code.

Prerequisites: Linux machine with ROS and rospy installed (not ROS2). Tested with Ubuntu 20.04.

# Install Vulkan Glasses (Linux)

- Verify that you have access to the Vulkan Glasses repository https://github.com/VIS4ROB-lab/vulkan_vrglasses_csv.
- Chose an installation folder. For instance, by running:

    ```bash
    SRC_FOLDER="/home/alain/src"
    ```

- Then, run:

    ```bash
    sudo apt install libeigen3-dev libvulkan-dev libgoogle-glog-dev libglm-dev glslang-tools libopencv-dev
    ```

- Clone the repository:

    ```bash
    cd $SRC_FOLDER
    git clone --recurse-submodules git@github.com:VIS4ROB-lab/vulkan_vrglasses_csv.git
    ```

- Open the file specified below. Then, uncomment the line `//outFragcolor.w = pushConsts.idobj;` and save the file.

    ```bash
    # For instance using vim
    vim $SRC_FOLDER/vulkan_vrglasses_csv/vrglasses_for_robots/shaders/vrglasses4robots_shader.frag
    ```

- Finally, install Vulkan:

    ```bash
    cd $SRC_FOLDER/vulkan_vrglasses_csv/vrglasses_for_robots/shaders/
    sh build_vrglasses4robots_shaders.sh

    cd $SRC_FOLDER/vulkan_vrglasses_csv/vrglasses_for_robots
    mkdir build

    cd $SRC_FOLDER/vulkan_vrglasses_csv/vrglasses_for_robots/build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8
    ```


# Install COLMAP 3.8 from source (Linux)

Note: The following commands are mostly equivalent to the instructions found on [https://colmap.github.io/install.html#installation](https://colmap.github.io/install.html#installation) .

- Install the dependencies:

    ```bash
    sudo apt-get install \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev
    ```

- Clone COLMAP and chose the 3.8 version

    ```bash
    cd $SRC_FOLDER
    git clone https://github.com/colmap/colmap.git
    cd colmap
    git checkout 3.8
    ```

- Install COLMAP:

    ```bash
    cd $SRC_FOLDER/colmap
    mkdir build
    cd build
    cmake .. -GNinja
    ninja
    sudo ninja install
    ```

- Verify that the installation succeeded by running:

    ```bash
    colmap -h
    ```

    Also verify that the installed version is  `3.8`.

    Note: if the above command did not succeed, you may have to add COLMAP to your PATH.


# Setup the semantic_ba_nature_alain repository (Linux)

### Repository

- Verify that you have access to the GitHub https://github.com/VIS4ROB-lab/semantic_ba_nature_alain repository.
- Chose an installation folder. For instance, use the same as before:

    ```bash
    SRC_FOLDER="/home/alain/src"
    ```

- Clone the repository, by running:

    ```bash
    cd $SRC_FOLDER
    git clone git@github.com:VIS4ROB-lab/semantic_ba_nature_alain.git
    ```


### Model resources

- Download the resources folder from this [Google Drive link](https://drive.google.com/drive/folders/1wYjGZfL6xuBpiCTf-19OXrkKw-ZKm-F6?usp=sharing).
- Save the content of the downloaded folder to `$SRC_FOLDER/semantic_ba_nature_alain/data/resources`.
- The `data/resources` folder should like this:

    ```bash
    data/resources
    ├── cube
    │   ├── cube_large.obj
    │   ├── cube.obj
    │   ├── texture_city.jpg
    │   ├── texture_color.jpg
    │   ├── texture_dice.jpg
    │   ├── texture_earth.jpg
    │   ├── texture_mountains.jpg
    │   └── texture_sea.jpg
    ├── cylinder
    │   ├── cylinder_large.obj
    │   ├── cylinder.obj
    │   ├── cylinder_tilted.obj
    │   └── texture_mesopotamia.jpg
    ├── grass
    │   ├── grass.jpg
    │   └── grass.obj
    ├── terrain
    │   ├── kastelhof.jpg
    │   └── kastelhof.obj
    └── tree
        ├── canopy.obj
        ├── texture_spring.jpg
        ├── tree_full.obj
        └── trunk.obj
    ```


### Adapt the Vulkan paths

- Open the `vk_glasses_csv_flags.txt` file under `$SRC_FOLDER/semantic_ba_nature_alain/data/config/vk_glasses_csv_flags.txt` and adapt all the paths that contain `alain`.
It is important to write the paths explicitly starting from the `/home/...` directory, as using `~/...` does not seem to work.

    ```bash
    --output_folder_path=/home/alain/src/semantic_ba_nature_alain/data/results/last_dataset/output
    --fx=2559
    --fy=2559
    --cx=1536
    --cy=1152
    --far=1000
    --near=0.10000000000000001
    --output_h=2304
    --output_w=3072
    --mesh_obj_file=
    --mesh_texture_file=
    --model_folder=/home/alain/src/semantic_ba_nature_alain/data/resources
    --model_list_file=/home/alain/src/semantic_ba_nature_alain/data/config/model_def_list.txt
    --model_pose_file=/home/alain/src/semantic_ba_nature_alain/data/config/model_poses_list.txt
    --ortho=false
    --pose_file=/home/alain/src/semantic_ba_nature_alain/data/config/image_poses.txt
    --resource_folder=/home/alain/src/semantic_ba_nature_alain/data/resources
    --shader_folder=/home/alain/src/vulkan_vrglasses_csv/vrglasses_for_robots/shaders
    --step_skip=1
    ```

- Open the `run_create_dataset` located at `$SRC_FOLDER/semantic_ba_nature_alain/run_create_dataset` file and adapt the `VK_FOLDER` path:

    ```bash
    # Installation or source code of vulkan glasses
    VK_FOLDER=/home/alain/src/vulkan_vrglasses_csv
    ```


### Install the python package

- Install the python package `sba-package` in editable mode (together with all its dependencies).

    ```bash
    pip install -e $SRC_FOLDER/semantic_ba_nature_alain
    ```

- Verify that the `sba-pacakge` was successfully installed by running the following command. This command should not raise any errors.

    ```bash
    cd
    python3 -c "import sba"
    ```


### Authorize the scripts for execution

- Run the following:

    ```bash
    cd $SRC_FOLDER/semantic_ba_nature_alain
    chmod +x run_* read_poses reconstruction_from_poses
    ```