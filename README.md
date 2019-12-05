# trt_pose_demo
A Demo Application for NVIDIA TensorRT Pose Estimation

## What does this application do?
This application just displays pose estimation results of camera-captured image with [NVIDIA TensorRT pose estimation](https://github.com/NVIDIA-AI-IOT/trt_pose).

## Prerequisites
- NVIDIA Jetson Nano/TX2/AGX Xavier Developer Kit with JetPack installed
- USB web camera or MIPI CSI camera like Raspberry Pi v2 camera

## Setup
1. Install PyTorch and torchvision. Please refer to [this page](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano-version-1-3-0-now-available/).
1. Install [NVIDIA-AI-IOT/torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt).
1. Install [NVIDIA-AI-IOT/trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose).
1. Install this application.
```
$ git clone https://github.com/MACNICA-CLAVIS-NV/trt_pose_demo
$ cd trt_pose_demo
```
1. Copy [the resnet18_baseline_att_224x224_A model file](https://github.com/NVIDIA-AI-IOT/trt_pose#models) to the application directory.
1. Copy [the task description file](https://github.com/NVIDIA-AI-IOT/trt_pose/blob/master/tasks/human_pose/human_pose.json) to the application directory.

## Usage
