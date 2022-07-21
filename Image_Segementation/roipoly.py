from roipoly import RoiPoly
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("nemo.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)

roi = RoiPoly(color='r')

# Define ROI of Blue barrel
mask = roi.get_mask(img[:, :, 0])
plt.imshow(mask)
# roi.display_roi()

plt.show()