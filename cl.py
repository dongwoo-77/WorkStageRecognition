import numpy as np
import cv2
import pyrealsense2 as rs
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import copy

# C onfiguredepth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        s = time.time()
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        img = torchvision.transforms.functional.to_tensor(color_image)
        img = img.unsqueeze(0)
        img = img.to(device)
        # print(img.shape)

        model = models.resnext50_32x4d()
        model.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
        model.load_state_dict(torch.load('/home/nam/exp/model/model_state_dict_1', map_location=device))
        model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(img)

        _, predicted = torch.max(output, 1)
        # print(output)

        phase = ['start', 'cpu', 'cpu-lam', 'lam', 'lam-ssd', 'ssd', 'end']

        # phase = ['end', 'ing']

        # tm = time.localtime()
        # # print(tm)
        # tm = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
        # print("time :", 1/(time.time() - start))

        # cv2.imwrite('images/{}_{}.png'.format(tm, i), color_image)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)
        # print(predicted[0])
        # if int(predicted[0])%2 == 0:
        #     print(phase[0])
        # if int(predicted[0])%2 == 1:
        #     print(phase[1])
        print(phase[predicted[0]])
        
        
finally:

    # Stop streaming
    pipeline.stop()