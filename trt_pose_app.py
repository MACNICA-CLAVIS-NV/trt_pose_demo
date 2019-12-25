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
from video_app_utils import ContinuousVideoProcess
import argparse

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


class PoseEstimationProcess(ContinuousVideoProcess):

    def __init__(self, args):
        super().__init__(args)
        model = PoseCaptureModel( \
            WIDTH, HEIGHT, MODEL_WEIGHTS, OPTIMIZED_MODEL, TASK_DESC)
        colorConv = ColorConvert(args.qsize, self.capture)
        resize = Resize(args.qsize, colorConv)
        preprocess = Preprocess(args.qsize, resize, model)    
        inference = Inference(args.qsize, preprocess, model)  
        postprocess = Postprocess(args.qsize, inference, model)
  
        
def main():
    # Parse the command line parameters
    parser = argparse.ArgumentParser(description='TRT Pose Demo')
    ContinuousVideoProcess.prepareArguments(parser)
    args = parser.parse_args()
    # Create continuous video process and start it
    vproc = PoseEstimationProcess(args)
    vproc.execute()
    

if __name__ == '__main__':
    main() 
    sys.exit(0)   
