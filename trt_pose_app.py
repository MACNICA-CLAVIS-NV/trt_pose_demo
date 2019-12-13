#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# MIT License
#
# Copyright (c) 2019 MACNICA Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE. 
#

import sys
import cv2
from pose_capture import PoseCaptureModel
from video_app_utils import PipelineWorker
from video_app_utils import ContinuousVideoCapture
from video_app_utils import IntervalCounter
import argparse

WINDOW_TITLE = 'NVIDIA Pose Estimation Model Demo'
WIDTH = 224
HEIGHT = 224
INPUT_RES = (WIDTH, HEIGHT)
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
TASK_DESC = 'human_pose.json'

class ColorConvert(PipelineWorker):
    
    def __init__(self, qsize, source):
        super().__init__(qsize, source)
        
    def process(self, srcData):
        orgFrame = srcData
        frame = cv2.cvtColor(orgFrame, cv2.COLOR_BGR2RGB)
        return (True, (frame, orgFrame))
        
class Resize(PipelineWorker):

    def __init__(self, qsize, source):
        super().__init__(qsize, source)
        
    def process(self, srcData):
        frame, orgFrame = srcData
        frame = cv2.resize(frame, INPUT_RES, interpolation=cv2.INTER_NEAREST)
        return (True, (frame, orgFrame))

class Preprocess(PipelineWorker):
    
    def __init__(self, qsize, source, model):
        super().__init__(qsize, source)
        self.model = model
        
    def process(self, srcData):
        frame, orgFrame = srcData
        frame = self.model.preprocess(frame)
        return (True, (frame, orgFrame))
        
class Inference(PipelineWorker):
    
    def __init__(self, qsize, source, model):
        super().__init__(qsize, source)
        self.model = model
        
    def process(self, srcData):
        frame, orgFrame = srcData
        cmap, paf = self.model.infer(frame)
        return (True, (cmap, paf, orgFrame))
        
class Postprocess(PipelineWorker):
    
    def __init__(self, qsize, source, model):
        super().__init__(qsize, source)
        self.model = model
        
    def process(self, srcData):
        cmap, paf, orgFrame = srcData
        self.model.postprocess(cmap, paf, orgFrame)
        return (True, orgFrame)

def main():
    # Parse the command line parameters
    parser = argparse.ArgumentParser(description='NVIDIA Pose Estimation Model Demo')
    parser.add_argument('--camera', '-c', \
        type=int, default=0, metavar='CAMERA_NUM', \
        help='Camera number, use any negative integer for MIPI-CSI camera')
    parser.add_argument('--width', \
        type=int, default=800, metavar='WIDTH', \
        help='Capture width')
    parser.add_argument('--height', \
        type=int, default=600, metavar='HEIGHT', \
        help='Capture height')
    parser.add_argument('--fps', \
        type=int, default=20, metavar='FPS', \
        help='Capture frame rate')
    parser.add_argument('--qsize', \
        type=int, default=2, metavar='QSIZE', \
        help='Capture queue size')
    parser.add_argument('--qinfo', \
        action='store_true', \
        help='If set, print queue status information')
    parser.add_argument('--mjpg', \
        action='store_true', \
        help='If set, capture video in motion jpeg format')
    args = parser.parse_args()
    
    fpsCounter = IntervalCounter(10)
    
    model = PoseCaptureModel( \
        WIDTH, HEIGHT, MODEL_WEIGHTS, OPTIMIZED_MODEL, TASK_DESC)
    
    fourcc = None
    if args.mjpg:
        fourcc = 'MJPG'
    capture = ContinuousVideoCapture( \
        args.camera, args.width, args.height, args.fps, args.qsize, fourcc) 
  
    colorConv = ColorConvert(args.qsize, capture)
    resize = Resize(args.qsize, colorConv)
    preprocess = Preprocess(args.qsize, resize, model)    
    inference = Inference(args.qsize, preprocess, model)  
    postprocess = Postprocess(args.qsize, inference, model)
    
    postprocess.start()
    inference.start()
    preprocess.start()
    resize.start()
    colorConv.start()
    capture.start()
    
    while True:
    
        frame = postprocess.get()
        
        # Debug
        if args.qinfo:
            print('%02d %06d - %02d %06d - %02d %06d - %02d %06d - %02d %06d - %02d %06d' % ( \
                capture.qsize(), capture.numDrops, \
                resize.qsize(), resize.numDrops, \
                colorConv.qsize(), colorConv.numDrops, \
                preprocess.qsize(), preprocess.numDrops, \
                inference.qsize(), inference.numDrops, \
                postprocess.qsize(), postprocess.numDrops
                ))
        
        interval = fpsCounter.measure()
        if interval is not None:
            fps = 1.0 / interval
            fpsInfo = '{0}{1:.2f}   ESC to Quit'.format('FPS:', fps)
            cv2.putText(frame, fpsInfo, (32, 32), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        
        cv2.imshow(WINDOW_TITLE, frame)
        
        # Check if ESC key is pressed to terminate this application
        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
            
    postprocess.stop()
    inference.stop()
    preprocess.stop()
    resize.stop()
    colorConv.stop()
    capture.stop()

if __name__ == '__main__':
    main() 
    sys.exit(0)   
