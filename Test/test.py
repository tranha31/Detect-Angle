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

image = cv2.imread('test611p3.png')
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
    # equations.append(np.cross([x1,y1,1],[x2,y2,1]))
  
    # x0 = (x1+x2)/2
    # y0 = (y1+y2)/2
    # print(x0, y0)
    # cv2.line(image,(x0,y0),(0,0),(255,0,0),2)

# for line in lines:
#     print(line)

result = list(filter(lambda x: math.atan(-x['line'][0]/x['line'][1])*(180/np.pi) <= 0, equations))

maxLine = max(result, key=lambda x:x['rho'])

x0 = (0, round(-maxLine['line'][2]/maxLine['line'][1]))
x1 = (round(-maxLine['line'][2]/maxLine['line'][0]), 0)

cv2.line(image, x0, x1, (255,0,0), 2)



# font = cv2.FONT_HERSHEY_SIMPLEX
# thetas = []
# N = len(equations)
# for ii in range(1,N):
#     a1,b1,c1 = equations[0]
#     a2,b2,c2 = equations[ii]
#     # intersection point
#     pt = np.cross([a1,b1,c1],[a2,b2,c2])
#     pt = np.int16(pt/pt[-1])
#     # angle between two lines
#     num = a1*b2 - b1*a2
#     den = a1*a2 + b1*b2
#     if den != 0:
#         theta = abs(np.arctan(num/den))*180/3.1416
#         # show angle and intersection point
#         cv2.circle(image, (pt[0],pt[1]), 5, (255,0,0), -1)
#         cv2.putText(image, str(round(theta, 1)), (pt[0]-20,pt[1]-20), font, 0.8, (255,0,0), 2, 0)
#         thetas.append(theta)

plt.imshow(image)
plt.show()