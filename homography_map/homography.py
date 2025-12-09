# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils
#import plotly.express as px
# %matplotlib inline


# Read image to be rectify
img1 = cv.imread('C:/Repositories/mobile_thread_device_robot/homography_map/raw.jpg')
img1a = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

#interactive selection of points
# plt.imshow(img1a)
# plt.title("Select four corners of the area to be rectified, starting from top-left and going clockwise")
# corners_img1 = plt.ginput(4)
# plt.show()
# corners_img1 = np.array(corners_img1)

# Define four points in the source image and their corresponding destination
# coordinates to estimate the homography for rectification.
corners_img1 = np.array([[210,63],[905,55],[944,3677],[205,3675]])
corners_img2 = np.array([[200,200],[939,200],[939,3822],[200,3822]])


src_pts = np.float32(corners_img1)
dst_pts = np.float32(corners_img2)

# Compute the homography and apply it to the image
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
img4 = cv.warpPerspective(img1a, M, (img1a.shape[1],img1a.shape[0]))

# Show the result plotting the source and destination points
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(1,2,1)
plt.imshow(img1a)
plt.plot(corners_img2[:,0],corners_img2[:,1],'*b')
plt.plot(corners_img1[:,0],corners_img1[:,1],'*r')
ax2 = fig.add_subplot(1,2,2)
plt.imshow(img4)
plt.plot(corners_img2[:,0],corners_img2[:,1],'*b')
plt.show()


# Save the rectified image
cv.imwrite('C:/Repositories/mobile_thread_device_robot/homography_map/rectified_image.jpg', cv.cvtColor(img4, cv.COLOR_RGB2BGR))
