import torch
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Step 1: installing dependencies.

# %%

import torch
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Step 2: Loading the model

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# 5. Learning to train model on a custom data set.
images_path = 'data/images'
labels = ['happy','angry']
number_imgs = 20

cap = cv2.VideoCapture(0)

for label in labels:
    print("Getting the training images for {}".format(label))
    time.sleep(5)

