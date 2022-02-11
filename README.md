# shape_control_DLO

[Project website](https://mingrui-yu.github.io/shape_control_DLO)

Code for ICRA 2022 paper "Shape Control of Deformable Linear Objects with Offline and Online Learning of Local Linear Deformation Models".

Here we provide:

* the training dataset
* the offline learned deformation model
* the controller
* the built simulation environment



## Dependencies

* Ubuntu 18.04
* ROS Melodic
* Nvidia driver & CUDA
* PyTorch in python3 env
* Unity for Linux 2020.03
* [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents)
* [PyTorch-Radial-Basis-Function-Layer](https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer): we use the code for the implementation of RBFN in pytorch



## Installation

* Install ROS Melodic on Ubuntu 18.04
* Install PyTorch with cuda support in python3 env
* Install Unity for Linux 2020.03 [doc](https://docs.unity3d.com/2020.2/Documentation/Manual/GettingStartedInstallingHub.html)
* Install Unity ML-Agents Toolkit [doc](https://github.com/Unity-Technologies/ml-agents/blob/release_18_docs/docs/Installation.md)

Install the following dependences in your python3 env:

```
pip install numpy
pip install matplotlib
pip install sklearn
pip install rospkg
pip install empy
pip install PyYAML
pip install mlagents==0.27.0
pip install gym
pip install gym_unity
pip install scipy
```

Clone the repository:

```
git clone https://github.com/Mingrui-Yu/shape_control_DLO.git
```

Build the catkin workspaces:

```
cd <YOUR_PATH>/shape_control_DLO/ws_dlo
catkin_make
```

Change the variable "project_dir" in *shape_control_DLO/'ws_dlo/src/dlo_system_pkg/config/params_sim.yaml* to '<YOUR_PATH>/shape_control_DLO/'.

If you want to retrain the offline model, you can download the training dataset: [OneDrive](https://1drv.ms/u/s!Aj5mBPrHs4TBhfMLYN1zGvSRpEaeiQ?e=aQDsWs). Then unzip it and put it in *<YOUR_PATH>/shape_control_DLO*/.



## Usage

Give permissions to the simulation environment:

```
chmod -R 755 <YOUR_PATH>/shape_control_DLO/env_dlo/env_2D
chmod -R 755 <YOUR_PATH>/shape_control_DLO/env_dlo/env_3D
```

Source the workspace:

```
cd <YOUR_PATH>/shape_control_DLO/ws_dlo
source devel/setup.bash
```

### Parameter settings

Modifiable parameters in *shape_control_DLO/ws_dlo/src/dlo_system_pkg/config/params_sim.yaml*:

* "project_dir": change it to your path to the project
* "env/dimension": '2D' or '3D'
* "controller/online_learning/lr_task_e": corresponding to $\eta_1$ in paper 
* "controller/online_learning/lr_approx_e": corresponding to $\eta_2$ in paper

### Run the control tasks

**Activate your python3 env**, and run:

```
roslaunch dlo_system_pkg simulation_run.launch
```

### Train the neural network

First, run

```
roslaunch dlo_system_pkg upload_params.launch
```

Then, **Activate your python3 env**, and run the python script:

```
python src/dlo_manipulation_pkg/scripts/RBF.py
```



## Details

### Built simulation environment

The simulation environment is an executable file built by Unity. In both 2D and 3D environment, the ends of the DLO are grasped by two grippers which can translate and rotate. 

Both the control rate and the data collection rate are 10 Hz.

The manipulated DLO is with a length of 0.5m and a diameter of 1cm.

#### Coordinate

We use a standard right-hand coordinate in the simulation environment.

In 2D tasks, the x axis is towards the bottom of the screen. The y axis is towards the right of the screen. The z axis is towards the outside of the screen.

In 3D tasks, the x axis is towards the outside of the screen. The y axis is towards the right of the screen. The z axis is towards the top of the screen.

Actually, the coordinates in 2D and 3D environment are the same. Only the position of the camera is changed.

#### Feature

Ten features are uniformly distributed along the DLO, which are represented by blue points. Note that the first and last features are the left and right ends, so they actually represent the positions of the robot end-effectors. 

#### Desired shape

The desired shape is represented by the desired positions of 10 features. All desired shapes are chosen from the training dataset, so they are ensured to be feasible. The 100 2D desired shapes are stored in *shape_control_DLO/env_dlo/env_2D_Data/StreamingAssets/desired_shape/2D/desired_positions.npy* and the 100 3D desired shapes are stored in *shape_control_DLO/env_dlo/env_2D_Data/StreamingAssets/desired_shape/3D/desired_positions.npy* .

Note that in control tasks, only the desired positions of the internal 8 features are considered (set as the target points).

#### State

The state is a 116-dimension vector.

* 0~29: the positions of the 10 features (10*3)
* 30~43: the pose of the two end-effectors
  * left end positions (3) + left end orientation (4) + right end position (3) + right end orientation (4)
  * The representation of the orientations is quaternion.
* 44~73: the velocities of the 10 features (10*3)
* 74~85: the velocities of the end-effectors
  * left end linear velocity (3) + left end angular velocity (3) + right end linear velocity (3) + right end angular velocity (3)
  * The representation of the angular velocities is rotation vector.
* 86~115: the desired positions of the 10 features (10*3)

Note that in 2D environment, the dimension of the position of one feature is still three, but the value in the z axis is always zero.

#### Action

The action is a 12-dimension vector.

* 0~2: the linear velocity of the left end effector
* 3~5: the angular velocity of the left end effector
* 6~8: the linear velocity of the right end effector
* 9~11: the angular velocity of the right end effector

In our implementation, the action is formulated as the above format in both 2D and 3D environment. Thus, in 2D tasks, we need to output valid control input in the controller script, where the [2, 3, 4, 8, 9, 10] dimension of the control input must be zero.

### Training dataset

Our offline collected data are in *shape_control_DLO/data/train_data*.

* state.py: the data of the manipulated DLO. It is not used for training the offline model, but only for extracting the feasible desired shapes and testing the performance of the network.
* state_1.npy ~ state_10.npy: the data of 10 different DLOs. They are used for training the offline model. We collect 18k (30min) data on each DLO.
* state_all.npy: the concatenation of all data of 10 different DLOs.
* state_part.npy: the concatenation of part data of 10 different DLOs. We choose 3k data on each DLO.

The format of the training dataset is (-1, 116). Each row is a 116-dimension state vector.

The lengths and diameters of the 10 DLOs:

| DLO  | length(m) | diameters(mm) |
| :--: | :-------: | :-----------: |
|  1   |    0.7    |      20       |
|  2   |    0.8    |       6       |
|  3   |    0.6    |      16       |
|  4   |    0.4    |       8       |
|  5   |    0.4    |      18       |
|  6   |    1.0    |      18       |
|  7   |    1.0    |      30       |
|  8   |    0.8    |      10       |
|  9   |    0.6    |       8       |
|  10  |    0.5    |      14       |

## Citation
Please cite our paper if you find it helpful :)

Before the official ICRA version being available online, you can cite the arXiv version:
```
@article{yu2021shape,
  title={Shape Control of Deformable Linear Objects with Offline and Online Learning of Local Linear Deformation Models},
  author={Yu, Mingrui and Zhong, Hanzhong and Li, Xiang},
  journal={arXiv preprint arXiv:2109.11091},
  year={2021}
}
```


## Contact

If you have any question, feel free to raise an issue (recommended) or contact the authors: Mingrui Yu, ymr20@mails.tsinghua.edu.cn