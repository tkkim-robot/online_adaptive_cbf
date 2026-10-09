# online_adaptive_cbf

This repository contains the implementation of an online adaptive framework for Control Barrier Functions (CBFs) in input-constrained nonlinear systems. The algorithm dynamically adapts CBF parameters to optimize performance while ensuring safety.


<div align="center">
<img src="https://github.com/user-attachments/assets/62b98ddd-1999-4a08-b68c-994f800ff8a2" width="700px">

<div align="center">

[[Homepage]](https://www.taekyung.me/online-adaptive-cbf)
[[Arxiv]](https://arxiv.org/abs/2409.14616)
[[Video]](https://youtu.be/255IUS1f6Lo)
[[Research Group]](https://dasc-lab.github.io/)

</div>
</div>


## Features

- Implementation of the Probabilistic Ensemble Neural Network ([PENN](https://github.com/tkkim-robot/online_adaptive_cbf/tree/main/nn_model/penn)) which offers parallelized inference without an outer for loop. The predicted output can be interpreted as a Gaussian Mixture Model (GMM). (see [Kim et al.](https://arxiv.org/abs/2305.12240))
- Measurement of [closed-form epistemic uncertainty](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/nn_model/penn/divergence/utility.py) from the PENN model's predictions. (see [Kim et al.](https://arxiv.org/abs/2305.12240))
- Integration with the [`safe_control`](https://github.com/tkkim-robot/safe_control) repository for simulating robotic navigation, offering various robot dynamics, controllers, and RGB-D type sensor simulation.
- Implementation of the Online Adaptive ICCBF, adapting ICCBF parameters online based on the robot's current state and nearby environment.


## Installation
To install this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone --recursive https://github.com/tkkim-robot/online_adaptive_cbf.git
   cd online_adaptive_cbf
   ```

   If you've already cloned the repository without the --recursive flag, you can initialize and update the submodules with:
   ```bash
   git submodule update --init --recursive
   ```

2. (Optional) Create and activate a virtual environment

3. Install the dependencies:
   ```bash
   python -m pip install -r requirements-cpu.lock
   ```
   For CUDA, use `requirements-jax.lock` instead and run the examples with `--device gpu`. Use CPU for bicycle.

4. Download the trained weights and calibration files:
   ```bash
   python -m oa_cbf_jax.model_release
   ```
   This downloads the [model release](https://github.com/tkkim-robot/online_adaptive_cbf/releases/tag/oa-cbf-models-2026-10) into `models/paper/`. Training datasets are not required to run the examples.


## Getting Started

Familiarize with APIs and examples with the scripts in [`online_adaptive_cbf.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/online_adaptive_cbf.py)

### Basic Example
You can run our test example by:

```bash
python online_adaptive_cbf.py
```

The default example runs the Quad2D navigation scenario using GAT to adapt the CBF parameters online with a CBF-QP controller.

You can also test with compared methods using `--method`:

- `ours_fc`: Online adaptation using the nearest-obstacle fully connected encoder.
- `fixed_low` and `fixed_high`: Fixed conservative and aggressive CBF parameters.
- `optimal_decay`: Optimal Decay MPC-CBF for unicycle and quadrotors ([reference](https://ieeexplore.ieee.org/document/9683174)), or Optimal Decay CBF-QP for bicycle.
- `optimal_decay_qp`: Optimal Decay CBF-QP for unicycle and Quad2D ([reference](https://ieeexplore.ieee.org/document/9482626)).
- `barriernet`: BarrierNet.

```bash
python online_adaptive_cbf.py --dynamics quad2d --method ours_fc
python online_adaptive_cbf.py --dynamics quad2d --method fixed_low
```

Example navigation results:

|     MPC-CBF w/ low parameters            |       MPC-CBF w/ high parameters     |
| :------------------: | :--------------------------: |
|  <img src="https://github.com/user-attachments/assets/6a67bf2d-0c0f-437f-8fc0-8d21511b9ab6"  height="170px"> | <img src="https://github.com/user-attachments/assets/8151d102-6fbe-4a93-8967-7004c8e0b2cb"  height="170px"> |

|     Optimal Decay CBF-QP  |       Optimal Decay MPC-CBF    |
| :---------------------: | :----------------------------: |
|  <img src="https://github.com/user-attachments/assets/e43f72bc-475a-403d-bac8-41a077acdaf1"  height="170px"> | <img src="https://github.com/user-attachments/assets/ae2ecb58-254b-4334-84d6-8c52508c9973"  height="170px"> |


|     Ours (Online Adaptive MPC-ICCBF)      |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/5d5806c1-31a9-42fb-806f-04ece91d54ba"  height="170px"> |

The green point is the goal location, and the gray circles are the obstacles that are known a priori.

### Live Simulation Preview

Use [`examples/run_simulation.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/examples/run_simulation.py) to preview the navigation scenarios:

```bash
python examples/run_simulation.py --list
python examples/run_simulation.py --dynamics quad2d --method ours_gat
python examples/run_simulation.py --dynamics dynamic_unicycle --method ours_fc --hold
```

Available dynamics are `unicycle`, `quad2d`, `quad3d`, and `bicycle`. Use `--list` to show the methods supported by each dynamics.

### Paper Media

> Warning: This feature requires a lot of computation time. For interactive visualization, use [`examples/run_simulation.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/examples/run_simulation.py)


Generate videos with [`examples/run_simulation.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/examples/run_simulation.py) (requires `ffmpeg`):

```bash
python examples/run_simulation.py --dynamics quad2d --method ours_gat --video paper_media/quad2d_gat.mp4 --headless
python examples/run_simulation.py --dynamics bicycle --method ours_fc --video paper_media/bicycle_fc.mp4 --headless
```

## Module Breakdown

### Safety Loss Density Function

The [`safety loss density function`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/safety_loss_function.py) is designed to quantify the collision risk between the robot and obstacles. This safety loss is computed based on the robot's state and the obstacles' locations.

|    Safety Loss    |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/fd040200-cbcb-4547-894e-8480d7495105"  height="350px"> |

### Data Generation

You can use [`data_generation.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/data_generation.py) to collect training dataset. It will store `risk_level` and `deadlock_time` as the ground truth. 

The `risk_level` refers to the maximum safety loss value recorded during the navigation (see the illustration below).

|    Safety Loss during Navigation     |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/591813c0-15e3-4857-8949-eef009a2697a"  height="250px"> |


### PENN Prediction

To train the PENN model, use the script [`penn/train_data.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/nn_model/train_data.py). An example of the prediction results after training is shown below: 

|    Mean Predicted Risk Level    |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/461e3568-890b-4b31-8739-7271560cf747"  height="350px"> |

You can observe that the predicted risk level becomes higher as the CBF parameter increases, the distance to the obstacle decreases, the velocity increases, and the relative angle to the obstacle becomes smaller.

### Distributionally Robust CVaR

<p align="left">
  <img src="https://github.com/user-attachments/assets/a55809d8-a027-4dfc-a215-eb92fe592d4f" height="200px">
</p>

Please refer to our repository [`cvar-gmm-filter`](https://github.com/signalkee/cvar-gmm-filter/tree/7405e05f7455f320b2c7b0ae72cef31a82d4a4f8) for more details.

### Visualize Prediction Results for CBF Parameters of Interest

[`plot/plot_realtime_gmm_predictions.py`](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/plot/plot_realtime_gmm_predictions.py) provides an online plotting tool to visualize the predicted GMM distribution of the candidate CBF parameters. Here is the example of visualizing the predicted `risk_level` with three candidates, without adapting the parameters.


|    Single Obstacle         |    Multiple Obstacles    |
| :------------------: | :--------------------------: |
|  <img src="https://github.com/user-attachments/assets/3a883e17-bda5-4719-a6ac-92104e0209ff"  height="250px"> | <img src="https://github.com/user-attachments/assets/f62659bd-9d4d-4ff1-9b9b-9f24f0dd85c7"  height="250px"> |



## Citing

If you find this repository useful, please consider citing our paper:

```
@inproceedings{kim2025learning, 
    author    = {Kim, Taekyung and Kee, Robin Inho and Panagou, Dimitra},
    title     = {Learning to Refine Input Constrained Control Barrier Functions via Uncertainty-Aware Online Parameter Adaptation}, 
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    shorttitle = {Online-Adaptive-CBF},
    year      = {2025}
}
```

Our paper with more theoretical analysis:
```
@inproceedings{kim2025learning, 
    author    = {Kim, Taekyung and Kee, Robin Inho and Panagou, Dimitra},
    title     = {Learning to Refine Input Constrained Control Barrier Functions via Uncertainty-Aware Online Parameter Adaptation}, 
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    shorttitle = {Online-Adaptive-CBF},
    year      = {2025}
}
```

## Related Works

Here are some related projects/codes that you might be interested:

- [Visibility-Aware RRT*](https://github.com/tkkim-robot/visibility-rrt): Safety-critical Global Path Planning (GPP) using Visibility Control Barrier Functions

- [UGV Experiments with ROS2](https://github.com/tkkim-robot/px4_ugv_exp): Environmental setup for rovers using PX4, ros2 humble, Vicon MoCap, and NVIDIA VSLAM + NvBlox
