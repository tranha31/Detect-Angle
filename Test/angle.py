from itertools import count
import cv2
from skimage.transform import hough_line, hough_line_peaks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

tLower = 70
tUpper = 200
blurSize = 7

# xác định cạnh
def sobelX(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGaussian = cv2.GaussianBlur(imgGray, (blurSize, blurSize), 0)
    sobelX = cv2.Sobel(imgGaussian, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    return sobelX

def sobelY(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGaussian = cv2.GaussianBlur(imgGray, (blurSize, blurSize), 0)
    sobelY = cv2.Sobel(imgGaussian, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    return sobelY

def sobelXY(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGaussian = cv2.GaussianBlur(imgGray, (blurSize, blurSize), 0)
    sobelXY = cv2.Sobel(imgGaussian, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    return sobelXY

img = cv2.imread('img3_add.png')
imageX = sobelX(img)

imageY = sobelY(img)

imageXY = sobelXY(img)



# im = cv2.cvtColor(imageXY, cv2.COLOR_RGB2GRAY)
# imageXY1 = cv2.GaussianBlur(imageXY, (blurSize, blurSize), 0)
# print(imageXY)
# imageXY2 = cv2.Canny(imageXY, tLower, tUpper)

# cv2.imshow('Sobel X', imageXY2)
# cv2.imshow("â", imageXY)

# cv2.waitKey(0)

edgexy = np.sqrt(imageXY**2)//30


for i in range(0, len(edgexy)):
    for j in range(0, len(edgexy[i])):
        if edgexy[i][j] != 0:
            edgexy[i][j] = 1
        else:
            edgexy[i][j] = 0


filt = np.ones((5, 5), dtype = int)

def erosion(img, fil):
    S = img.shape
    F = fil.shape
    R = S[0] + F[0] - 1
    C = S[1] + F[1] - 1
    N = np.zeros((R, C))

    for i in range(S[0]):
        for j in range(S[1]):
            N[i+1, j+1] = img[i, j]
    
    for i in range(S[0]):
        for j in range(S[1]):
            k = N[i:i+F[0], j:j+F[1]]
            result = (k == fil)
            final = np.all(result == True)
            if final:
                img[i, j] = 1
            else:
                img[i, j] = 0
    
    return img

def dilation(img, fil):
    S = img.shape
    F = fil.shape
    R = S[0] + F[0] - 1
    C = S[1] + F[1] - 1
    N = np.zeros((R, C))

    for i in range(S[0]):
        for j in range(S[1]):
            N[i+1, j+1] = img[i, j]
    
    for i in range(S[0]):
        for j in range(S[1]):
            k = N[i:i+F[0], j:j+F[1]]
            result = (k == fil)
            final = np.any(result == True)
            if final:
                img[i, j] = 1
            else:
                img[i, j] = 0
    
    return img

def closing(img, fil):
    c1 = dilation(img, fil)
    c2 = erosion(c1, fil)
    return c2

def opening(img, fil):
    o1 = erosion(img, fil)
    o2 = dilation(o1, fil)
    return o2


closingImg = closing(edgexy, filt)

# filt1 = np.ones((3, 3), dtype = int)
openingImg = opening(closingImg, filt)

cv2.imshow('Sobel X', openingImg)
cv2.waitKey(0)

# finalImage = openingImg*255

# # cv2.imshow("test", finalImage)

# cv2.imwrite("test611p2.png", finalImage)


# image = cv2.imread('test611.png')
# imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# imageXY1 = cv2.GaussianBlur(imgGray, (blurSize, blurSize), 0)
# imageXY2 = cv2.Canny(imageXY1, 70, 200)

# lines = cv2.HoughLinesP(imageXY2,1,np.pi/180,70,60,10)

# print(len(lines))

# cv2.imshow('Sobel X', imageXY2)
# cv2.waitKey(0)

# cv2.imwrite("test.png", edgexy)

# equations = []
# for line in lines:
#     x1,y1,x2,y2 = line[0]
#     equations.append(np.cross([x1,y1,1],[x2,y2,1]))
#     cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)

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

# plt.imshow(image)
# plt.show()