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

'''A collection of utility classes for video applications.
'''

import queue
import threading
import cv2
import time
import numpy as np


class PipelineWorker():
    '''A worker thread for a stage in a software pipeline.
    This class is an abstruct class. Sub classes inherited from this class
    should implement a process stage of a software pipeline.
    
    +------------------+ +------------------+ +------------------+
    | PipelineWorker#1 | | PipelineWorker#2 | | PipelineWorker#3 |<-get()
    | getData()->(Q)-----> process()->(Q)-----> process()->(Q)----->
    +------------------+ +------------------+ +------------------+
    
    Attributes:
        queue: Queue to store outputs processed by this instance.
        source: Data source (assumped to be other PipelineWorker instance) 
        sem: Semaphore to lock this instance.
        flag: If ture, the processing loop is running.
        numDrops: Total number of dropped outputs.
        thread: Worker thread runs the _run instance method.
    '''
    
    def __init__(self, qsize, source=None):
        '''
        Args:
            qsize(int): Output queue capacity
            source(PipelineWorker): Data source. If ommited, derived class
                should implement the getData method.
        '''
        self.queue = queue.Queue(qsize)
        self.source = source
        self.sem = threading.Semaphore(1)
        self.flag = False
        self.numDrops = 0
    
    def __del__(self):
        pass
        
    def process(self, srcData):
        '''Data processing(producing) method called in thread loop.
        Derived class should implemens this method.
        
        Args:
            srcData: Source data 
        '''
        return (True, None)
        
    def getData(self):
        '''Returns a output to data consumer.
        '''
        if self.source is None:
            return None
        else:
            return self.source.get()
        
    def _run(self):
        with self.sem:
            self.flag = True
        while True:
            with self.sem:
                if self.flag == False:
                    break
            src = self.getData()
            ret, dat = self.process(src)
            if ret == False:
                break
            if self.queue.full():
                self.queue.get(block=True)
                self.numDrops += 1
            self.queue.put(dat)
               
    def start(self): 
        '''Starts the worker thread.
        '''     
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        
    def get(self):
        '''Gets a output.
        '''
        return self.queue.get(block=True)
        
    def stop(self):
        '''Stops the worker thread.
        '''
        with self.sem:
            self.flag = False
        self.thread.join()
        
    def qsize(self):
        '''Returns the number of the current queued outputs
        '''
        sz = 0
        with self.sem:
            sz = self.queue.qsize()
        return sz
        

class ContinuousVideoCapture(PipelineWorker):
    '''Video capture workeer thread
    '''

    GST_STR_CSI = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 \
    ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
    
    def __init__(self, cameraId, width, height, fps=30, qsize=30, fourcc=None):
        '''
            Args:
                cameraId(int): Camera device ID, if negative number specified,
                    the CSI camera will be selected.
                width(int): Capture width
                height(int): Capture height
                fps(int): Frame rate
                qsize(int): Capture queue capacity
                fourcc(str): Capture format FOURCC string
        '''
        
        super().__init__(qsize)
        
        if cameraId < 0:
            # CSI camera
            gstCmd = ContinuousVideoCapture.GST_STR_CSI \
                % (width, height, fps, width, height)
            self.capture = cv2.VideoCapture(gstCmd, cv2.CAP_GSTREAMER)
            if self.capture.isOpened() is False:
                raise OSError('CSI camera could not be opened.')
        else:
            # USB camera
            # Open the camera device
            self.capture = cv2.VideoCapture(cameraId)
            if self.capture.isOpened() is False:
                raise OSError('Camera %d could not be opened.' % (cameraId))
                
            # Set the capture parameters
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fourcc is not None:
                self.capture.set(cv2.CAP_PROP_FOURCC, \
                    cv2.VideoWriter_fourcc(*fourcc))
            self.capture.set(cv2.CAP_PROP_FPS, fps)
        
         # Get the actual frame size
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def __del__(self):
        super().__del__()
        self.capture.release()
        
    def getData(self):
        ret, frame = self.capture.read()
        return (ret, frame)
        
    def process(self, srcData):
        return srcData


class IntervalCounter():
    '''A counter to measure the interval between the measure method calls.
    
    Attributes:
        numSamples: Number of samples to calculate the average.
        samples: Buffer to store the last N intervals.
        lastTime: Last time stamp
        count: Total counts
    '''

    def __init__(self, numSamples):
        '''
        Args:
            numSamples(int): Number of samples to calculate the average.
        '''
        self.numSamples = numSamples
        self.samples = np.zeros(self.numSamples)
        self.lastTime = time.time()
        self.count = 0
        
    def __del__(self):
        pass
        
    def measure(self):
        '''Measure the interval from the last call.
        
        Returns:
            The interval time count in second.
            If the number timestamps captured in less than numSamples,
            None will be returned.
        '''
        curTime = time.time()
        elapsedTime = curTime - self.lastTime
        self.lastTime = curTime
        self.samples = np.append(self.samples, elapsedTime)
        self.samples = np.delete(self.samples, 0)
        self.count += 1
        if self.count > self.numSamples:
            return np.average(self.samples)
        else:
            return None

