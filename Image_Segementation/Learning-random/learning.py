# file just for learning and experimenting with image segmentation.

# starting off with Region based segmentation.
from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

image = plt.imread('barrel-pics/2.14.png')

plt.imshow(image)

gray = rgb2gray(image)

gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0

print("hello")
gray = gray_r.reshape(gray.shape[0], gray.shape[1])
plt.imshow(gray, cmap='gray')

plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
