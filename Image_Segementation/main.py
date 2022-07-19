import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

barrel2_3 = cv2.imread('barrel-pics/2.3.png')
barrel2_3 = cv2.cvtColor(barrel2_3,cv2.COLOR_BGR2RGB)
plt.imshow(barrel2_3)
#%%
r, g, b = cv2.split(barrel2_3)
fig = plt.figure()
axis = fig.add_subplot(1,1,1, projection = "3d")
pixel_colors = barrel2_3.reshape((np.shape(barrel2_3)[0]*np.shape(barrel2_3)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()