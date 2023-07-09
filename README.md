# Object Manipulation Tasks in ObjectFolder Benchmark

This repository contains the implementation for the four object manipulation tasks in *The ObjectFolder Benchmark: Multisensory Learning with Neural and Real Objects*, including dynamic pushing, grasp stability prediction, contact refinement, and surface traversal.

![demo_screwdriver](/media/object_manipulation.png)

## Installation

You can manually clone the repository and install the package using:

```bash
git clone git@github.com:objectfolder/robotic_tasks.git
cd robotic_tasks
pip install -e .
```

To install dependencies:

```bash
pip install -r requirements.txt
```

## Content

This repo contains:

1) GelSight 1.5 simulation renderer
2) Mesh models and urdf files for GelSight, Panda robot arm, and objects.
3) Data collection pipeline for four robotic tasks
4) Training and testing pipeline for four robotic tasks

## Usage

### Grasp Stability Prediction Task

For grasp stability prediction task, we need to first collect training and test data. Please change the running file name in data_collection/scripts/data_collect.sh to grasp_stability.py

```bash
cd data_collection/script
./data_collect.sh
```

Then, we can use the collected data to train and evaluate the model performance.

```bash
cd grasp_stability_task
./run_grasp.sh
```
### Contact Refinement and Surface Traversal Tasks

For contact refinement and surface traversal tasks, they are using the same pipeline, we need to first collect training and test data for video prediction model. Please change the running file name in OBJ_Robot/robotic_tasks/data_collection/scripts/data_collect.sh accordingly.

```bash
cd data_collection/script
./data_collect.sh
```

Then, we can use the collected data to train and evaluate the video prediction model. We are using SVG' as our video prediction model. Implementation of this model is adopted from this [repo](https://github.com/s-tian/svg-prime).

```bash
cd refine_traversal_tasks/video_prediction_model
./train_#####.sh 
```

To evaluate the performance of video prediction model on the task, we adopt model predictive control as our policy.


```bash
cd refine_traversal_tasks/model_predictive_control
./run_ctrl_####.sh 
```

### Dynamic Pushing Task

For dynamic pushing task, we need to first collect training data and testing data. Please change the running file name in data_collection/scripts/data_collect.sh to dynamic_pushing.py. Note that the testing data will be 4 push trails from unseen objects. They will be used in the evaluation of the dynamic model.

```bash
cd data_collection/script
./data_collect.sh
```

Then, we can use the collected data to train the dynamic model.

```bash
cd dynamic_pushing_task/scripts
./train_dyna.sh
```

After we have the dynamic model, we can launch experiments to evaluate its performance. The pipeline is to collect very few push trails from the unseen object and use the dynamic model to learn a representation. This representation will be used in the model predictive control policy.

```bash
cd dynamic_pushing_task/scripts
./launch_exp.sh
```

## Operating System

We recommend running experiments on **Ubuntu**. The code and environment are tested using a Ubuntu 22.04 PC with a NVIDIA GTX 1080Ti Graphics Card.


