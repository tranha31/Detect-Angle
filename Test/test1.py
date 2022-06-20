from itertools import count
import cv2
from skimage.transform import hough_line, hough_line_peaks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

tLower = 70
tUpper = 200
blurSize = 7

image = cv2.imread('img3_bruit_phone.png')
imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imageXY1 = cv2.GaussianBlur(imgGray, (blurSize, blurSize), 0)
imageXY2 = cv2.Canny(imageXY1, 60, 100)

# cv2.imshow('Sobel X', imageXY2)
# cv2.waitKey(0)

lines = cv2.HoughLinesP(imageXY2,1,np.pi/180,100,190,10)

print(len(lines))


equations = []


for line in lines:
    x1,y1,x2,y2 = line[0]
    rho = line[0][0]
    theta = line[0][1]
    lineFunc = np.cross([x1,y1,1],[x2,y2,1])

    equations.append(
        {'rho': rho, 'theta': theta, 'line': lineFunc, 'd1': (x1, y1), 'd2': (x2, y2)}
    )


result = list(filter(lambda x: math.atan(-x['line'][0]/x['line'][1])*(180/np.pi) <= 0, equations))

maxLine = min(result, key=lambda x:x['rho'])

x0 = (0, round(-maxLine['line'][2]/maxLine['line'][1]))
x1 = (round(-maxLine['line'][2]/maxLine['line'][0]), 0)

img = cv2.imread("img3_bruit.png")

cv2.line(img, x0, x1, (255,0,0), 2)
cv2.line(image, x0, x1, (255,0,0), 2)

# plt.imshow(img)

# plt.imshow(image)
# plt.show()

cv2.imshow("origin", img)
cv2.imshow("detect", image)

cv2.waitKey(0)