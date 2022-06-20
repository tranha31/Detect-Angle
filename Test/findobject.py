from itertools import count
import cv2
from cv2 import findContours
from skimage.transform import hough_line, hough_line_peaks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

tLower = 70
tUpper = 200
blurSize = 7

image = cv2.imread('test611p2.png')
imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imageXY1 = cv2.GaussianBlur(imgGray, (blurSize, blurSize), 0)
imageXY2 = cv2.Canny(imageXY1, 60, 100)

kernel = np.ones((5,5), dtype=int)
imgDial = cv2.dilate(imageXY2, kernel=kernel, iterations=3)
imgThree = cv2.erode(imgDial, kernel=kernel, iterations=2)

contours, _ = cv2.findContours(imgThree,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

findContour = []
for c in contours:
    area = cv2.contourArea(c)
    if area > 10000:
        findContour.append(c)

stencil = np.zeros(image.shape).astype(image.dtype)
color = [255, 255, 255]
result = cv2.fillPoly(stencil, findContour[1], color)

# print(image)
# print("-----------")
# print(result)

cv2.imshow("ss", result)
cv2.waitKey(0)

# cv2.drawContours(image, findContour[1], -1, (0, 255, 0), 2)

# cv2.imwrite("result.png", result)

# print(contours)

# c=max(contours,key=cv2.contourArea)
# x,y,w,h = cv2.boundingRect(c)
# cv2.rectangle(imgGray,(x,y),(x+w,y+h),(125,125,125),2)