from itertools import count
import cv2
from skimage.transform import hough_line, hough_line_peaks
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

tLower = 70
tUpper = 200
blurSize = 7

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

image = cv2.imread("result.png")
imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imgGaussian = cv2.GaussianBlur(imgGray, (blurSize, blurSize), 0)

filt = np.ones((11, 11), dtype = int)

closingImg = closing(imgGaussian, filt)

# filt1 = np.ones((3, 3), dtype = int)
# openingImg = opening(closingImg, filt)


finalImage = closingImg*255

img = cv2.imread("img3_add.png")
cv2.imshow("origin", img)
cv2.imshow("ttt", finalImage)
cv2.waitKey(0)

# cv2.imwrite("test1162022.png", finalImage)
