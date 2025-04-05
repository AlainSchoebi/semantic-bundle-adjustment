# Use Semantic BAs (Windows)

The following instructions describe how to run the pixelwise semantic BA and the geometric semantic BA on Windows.

Prerequisites: Windows setup successfully completed ([Setup (Windows)](setup_windows.md)). Some dataset ready to use in the appropriate format can be useful (see [Prepare Dataset (Linux)](prepare_dataset_linux.md)).

# Prerequisites

- The following instructions all use the `Release` mode since it is significantly faster than the `Debug` mode.
- Ensure that the Release mode has been built in Visual Studio. If not, select `Release` and click on the green run button, as described in [Setup (Windows)](setup_windows.md).
- Specify the location of the COLMAP executable file and of the result folder. The result folder contains the datasets and can freely be chosen. For instance, run the following:

    ```bash
    COLMAP="C:\Users\alain\source\repos\colmap_semantic_ba_nature_alain\build\src\exe\Release\colmap.exe"
    RESULTS="C:\Users\alain\data\results"
    ```


# Pixelwise Semantic BA Usage

![psba.gif](psba.gif)

- The pixelwise semantic BA is referred to as the `semantic_bundle_adjuster` in the implementation.
- To display the available command arguments, run:

    ```bash
    $COLMAP semantic_bundle_adjuster -h
    ```

- The important arguments are shown here:

    ```bash
    $COLMAP semantic_bundle_adjuster \
    --input_path <path> \
    --output_path <path> \
    --SemanticBundleAdjustment.data_path <path> \
    --SemanticBundleAdjustment.numeric_relative_step_size <value> \
    --SemanticBundleAdjustment.depth_error_threshold <value> \
    --SemanticBundleAdjustment.error_computation_pixel_step <value>
    ```

    - The input path must be a folder containing a COLMAP model.
    - The output folder can freely be chosen but the folder must already exist. The output folder will contain the refined COLMAP model as well as some additional information.
    - The data path must be a folder with the `.h5` files containing the color, depth and semantic images for the model.
    - The numeric relative step size is the relative step size used for numerical differentiation. The description from the Ceres Solver can be found [here](https://github.com/ceres-solver/ceres-solver/blob/62c03d6ff3b1735d4b0a88006486cfde39f10063/include/ceres/numeric_diff_options.h#L43C1-L46C36).
    - The depth error threshold argument is the threshold defined as $d_\text{error}$ in the report.
    - The error computation pixel step determines the space between every pixel used for the semantic error computation.

# Geometric Semantic BA Usage

![tree_localization.gif](tree_localization.gif)

- The geometric semantic BA is referred to as `geometric_semantic_bundle_adjuster` in the implementation.
- To display the available command arguments, run:

    ```bash
    $COLMAP geometric_semantic_bundle_adjuster -h
    ```

- The important arguments are shown here:

    ```bash
    $COLMAP geometric_semantic_bundle_adjuster \
    --input_path <path> \
    --output_path <path> \
    --GeometricSemanticBundleAdjustment.data_path <path> \
    --GeometricSemanticBundleAdjustment.trunk_semantic_class <value> \
    --GeometricSemanticBundleAdjustment.input_geometry <path> \
    --GeometricSemanticBundleAdjustment.numeric_relative_step_size <value> \
    --GeometricSemanticBundleAdjustment.cylinder_parametrization <default|by_2_points> \
    --GeometricSemanticBundleAdjustment.refine_extrinsics <boolean> \
    --GeometricSemanticBundleAdjustment.refine_geometry <boolean> \
    --GeometricSemanticBundleAdjustment.include_landmark_error <boolean> \
    --GeometricSemanticBundleAdjustment.landmark_error_weight <value>
    ```

    - The trunk semantic class argument represents the value of the semantic class of the trunk. By default it is `250`.
    - The input geometry argument must be a text file containing the description of the initial cylinder in the following format:

        ```
        q 0.825156 0.0142376 0.0356231 0.563736 t -5.06639 -4.97302 1.76895 r 0.243773 h 2.30341
        ```

    - The cylinder parametrization argument determines which parametrization, *default* or *by_2_points,* is used in the optimization.
    - The refine extrinsics boolean indicates whether or not the camera poses are being optimized. Note that optimizing over the intrinsic parameters of the cameras is not supported.
    - The refine geometry boolean indicates whether or not the cylinder is being optimized.
    - The include landmark boolean indicates whether or not the 3D landmarks also contribute to the cost function or not. This would imply a mix between a usual BA and this geometric semantic BA.
    - The landmark error weight argument defines the weight associated with the landmark error. Refer to the code for a more precise definition of this weight.

# Example Experiments

Here are two example experiments to illustrate the two semantic BAs.

### Example Datasets

- On the Windows machine, download the example datasets from this [Google Drive Folder](https://drive.google.com/drive/folders/1LDblyztFUeJJ7hcVWMzgebDAgLPMMr2N?usp=sharing) and place them in the previously chosen `$RESULTS` folder.
- The `$RESULTS` folder should look like this:

    ```bash
    $RESULTS
    ├───tree_full
    │   │   image_poses.txt
    │   │   model_def_list.txt
    │   │   model_poses_list.txt
    │   │
    │   ├───colmap
    │   │   │   ...
    │   │
    │   ├───colmap_geo
    │   │   │   ...
    │   │
    │   ├───colmap_gt
    │   │   │   ...
    │   │
    │   ├───exp_1
    │   └───output
    │       │   ...
    │
    └───tree_split
        │   image_poses.txt
        │   model_def_list.txt
        │   model_poses_list.txt
        │
        ├───colmap
        │   │   ...
        │
        ├───colmap_geo
        │   │   ...
        │
        ├───colmap_gt
        │   │   ...
        │
        ├───cylinder_init
        │       a.txt
        │       b.txt
        │
        ├───exp_2
        └───output
            │   ...
    ```


## Experiment 1 - Pixelwise Semantic BA

This experiment uses the pixelwise semantic BA to try to improve the camera poses after they have been estimated via COLMAP. Note that the reconstructed camera poses from COLMAP have been geo-registered and saved to the `colmap_geo` folder.

- Run the following command to run the BA:

    ```bash
    $COLMAP semantic_bundle_adjuster \
    --input_path $RESULTS"/tree_full/colmap_geo" \
    --output_path $RESULTS"/tree_full/exp_1" \
    --SemanticBundleAdjustment.data_path $RESULTS"/tree_full/output" \
    --SemanticBundleAdjustment.numeric_relative_step_size 2e-3 \
    --SemanticBundleAdjustment.depth_error_threshold 1.5 \
    --SemanticBundleAdjustment.error_computation_pixel_step 5 \
    --SemanticBundleAdjustment.export_csv false
    ```

    Note: The optimization is expected to stop after 24 iterations, i.e. the last iteration is iteration 23.

- For the evaluation and visualization, switch to the Linux machine and transfer the files somehow.
- Place the datasets in the `semantic_ba_nature_alain/data/results` folder of the repository.
- Move to the `semantic_ba_nature_alain` repository, by for instance running:

    ```bash
    cd ~/src/semantic_ba_nature_alain
    ```

- To evaluate the reconstructed camera poses (`colmap_geo`) with the GT camera poses (`colmap_gt`) run:

    ```bash
    ./run_model_evaluation \
    data/results/tree_full/colmap_geo \
    data/results/tree_full/colmap_gt
    ```

    This will generate an output ROS bag saved at `out/last_model_evaluation.bag`.

- To evaluate the camera poses after the pixelwise semantic BA (`exp_1`) with the GT camera poses (`colmap_gt`) run:

    ```bash
    ./run_model_evaluation \
    data/results/tree_full/exp_1 \
    data/results/tree_full/colmap_gt
    ```

    This will generate an output ROS bag saved at `out/last_model_evaluation.bag` (this overwrites the previous one).

- If everything went alright, the camera pose error should have decreased from `0.0243m, 0.275°` to `0.0166m, 0.216°`.
- The different camera poses can be visualized in RViz. However, since the estimated and the refined camera poses are so close to the ground truth camera poses, it is very difficult to see any difference. To visualize the camera poses, use the following commands in separate terminals:

    ```bash
    roscore
    ```

    ```bash
    rviz -d config/rviz/config.rviz
    ```

    ```bash
    rosbag play out/last_model_evaluation.bag
    ```


## Experiment 2 - Trunk localization

This experiment uses the geometric semantic BA to locate the trunk of the tree using a cylinder. Note that the dataset must contain a specific class for the tree trunk (in our case class `250` as it is the default value). In this experiment, the camera poses are fixed and only the cylinder is being optimized.

- Run the following command to localize the trunk using the BA:

    ```bash
    $COLMAP geometric_semantic_bundle_adjuster \
    --input_path $RESULTS"/tree_split/colmap_gt" \
    --output_path $RESULTS"/tree_split/exp_2" \
    --GeometricSemanticBundleAdjustment.data_path $RESULTS"/tree_split/output" \
    --GeometricSemanticBundleAdjustment.input_geometry $RESULTS"/tree_split/cylinder_init/a.txt" \
    --GeometricSemanticBundleAdjustment.numeric_relative_step_size 1e-3 \
    --GeometricSemanticBundleAdjustment.cylinder_parametrization "default" \
    --GeometricSemanticBundleAdjustment.refine_extrinsics false \
    --GeometricSemanticBundleAdjustment.refine_geometry true
    ```

    Note: the optimization is expected to end with a mean IoU of `0.744`.

- Now switch to the Linux machine and transfer the files somehow.
- Place the datasets in the `semantic_ba_nature_alain/data/results` folder of the repository.
- Move to the `semantic_ba_nature_alain` repository, by for instance running:

    ```bash
    cd ~/src/semantic_ba_nature_alain
    ```

- To visualize the optimization, use the following command:

    ```bash
    python3 sba/visualization/vis_run_poses.py \
    --folder data/results/tree_split/exp_2
    ```

    This will generate an output ROS bag saved at `out/last_optim_steps.bag`.

- Then run the following in separate terminals:

    ```bash
    roscore
    ```

    ```bash
    rviz -d config/rviz/config.rviz
    ```

    ```bash
    rosbag play out/last_optim_steps.bag
    ```

    Note: to visualize the optimization in a loop use: `rosbag play out/last_optim_steps.bag -l`.

- If you want to visualize the 3D reconstruction at the same time, run the following:

    ```bash
    python3 sba/reconstruction_3d.py \
    --workspace_folder data/results/tree_split/ \
    --no-save_images
    ```

    ```bash
    rosbag play out/last_3d_reconstruction.bag
    ```

- Finally, one can run the BA again and experiment with other camera poses (`colmap_gt` or `colmap_geo`), or other cylinder parametrizations (`default` or `by_2_points`), or other initial cylinders (e.g. `$RESULTS"/tree_split/cylinder_init/a.txt` or `$RESULTS"/tree_split/cylinder_init/b.txt`).