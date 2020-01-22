# trt_pose_demo
A Demo Application for NVIDIA TensorRT Pose Estimation

## What does this application do?
This application just displays pose estimation results for camera-captured video or H.264/H.265 video file with [NVIDIA TensorRT pose estimation](https://github.com/NVIDIA-AI-IOT/trt_pose). This application also has a capability to save the results in a CSV file.

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
1. Copy [the human_pose.json task description file](https://github.com/NVIDIA-AI-IOT/trt_pose/blob/master/tasks/human_pose/human_pose.json) to the application directory.

## Usage
First, clock up your Jetson.
```
$ sudo nvpmodel -m 0
$ sudo jetson_clocks
```
The following command starts this application.
```
$ python3 trt_pose_app.py [-h] [--camera CAMERA_NUM] [--width WIDTH]
                       [--height HEIGHT] [--fps FPS] [--qsize QSIZE] [--qinfo]
                       [--mjpg] [--title TITLE] [--nodrop] [--repeat] [--h265]
                       [--model MODEL] [--task TASK_DESC] [--csv MAX_CSV_REC]
                       [--csvpath CSV_PATH] [--verbose]
                       [SRC_FILE]

TRT Pose Demo

positional arguments:
  SRC_FILE              Source video file

optional arguments:
  -h, --help            show this help message and exit
  --camera CAMERA_NUM, -c CAMERA_NUM
                        Camera number, use any negative integer for MIPI-CSI
  --width WIDTH         Capture width
  --height HEIGHT       Capture height
  --fps FPS             Capture frame rate
  --qsize QSIZE         Capture queue size
  --qinfo               If set, print queue status information
  --mjpg                If set, capture video in motion jpeg format
  --title TITLE         Window title
  --nodrop              If set, disable frame drop feature
  --repeat              If set, repeat video decoding
  --h265                If set, the specified video file will be assumed as
                        H.265. Otherwise, assumed as H.264
  --model MODEL         Model weight file
  --task TASK_DESC      Task description file
  --csv MAX_CSV_REC     Maximum CSV records
  --csvpath CSV_PATH    Directory path to save CSV files
  --verbose             If set, print debug message

```
For MIPI-CSI camera, use any negative number as the camera number.
```
$ python3 trt_pose_app.py --camera -1 
```
For USB Web camera, if you camera is detected as /dev/video1, use 1 as the camera number.
```
$ python3 trt_pose_app.py --camera 1
```
To get a CSV output, please use the **--csv** option with the maximum capture frames. You can specify the directory to hold the CSV file with the **--csvpath** option.
```
$ python3 trt_pose_app.py --camera 0 --csv 1000 --csvpath ./logs
```
To use the densenet121_baseline_att_256x256_B_epoch_160.pth pre-trained model which is also released at [the resnet18_baseline_att_224x224_A model file](https://github.com/NVIDIA-AI-IOT/trt_pose#models), use the **--model** option.
```
$ python3 trt_pose_app.py --camera 0 --model densenet121_baseline_att_256x256_B_epoch_160.pth
```
This application can accept not only camera capture but also H.264 movie file as input. For high resolution movie file input, the **--nodrop** option might be needed.
```
$ python3 trt_pose_app.py --nodrop test.mov
```
