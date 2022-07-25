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

# 3. Making Detections with Image.

img = 'https://ultralytics.com/images/zidane.jpg'

result = model(img)
print(result)

plt.imshow(np.squeeze(result.render()))

# 4. Real Time Detections

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
