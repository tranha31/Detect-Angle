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

img = cv2.imread("img4.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imageBlur = cv2.GaussianBlur(imgGray, (blurSize, blurSize), 0)
imageCanny = cv2.Canny(imageBlur, 60, 100)

kernel = np.ones((5,5), dtype=int)
imgDial = cv2.dilate(imageCanny, kernel=kernel, iterations=3)
imgThree = cv2.erode(imgDial, kernel=kernel, iterations=2)


contours, _ = cv2.findContours(imgThree,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

findContour = []
for c in contours:
    area = cv2.contourArea(c)
    if area > 4000:
        findContour.append(c)

print(len(findContour))

stencil = np.zeros(img.shape).astype(img.dtype)
color = [255, 255, 255]
filt = np.ones((11, 11), dtype = int)

phoneObject = cv2.fillPoly(stencil, findContour[2], color)
imgGray0 = cv2.cvtColor(phoneObject, cv2.COLOR_BGR2GRAY)
imgGaussian = cv2.GaussianBlur(imgGray0, (blurSize, blurSize), 0)
closingImg = closing(imgGaussian, filt)
phone = closingImg*255

# remoteObject = cv2.fillPoly(stencil, findContour[1], color)
# imgGray1 = cv2.cvtColor(remoteObject, cv2.COLOR_BGR2GRAY)
# imgGaussian1 = cv2.GaussianBlur(imgGray1, (blurSize, blurSize), 0)
# closingImg1 = closing(imgGaussian1, filt)
# remote = closingImg1*255

cv2.imshow("phone", phone)
cv2.waitKey(0)

# cv2.imshow("phone", phone)
# cv2.waitKey(0)
# cv2.imwrite("img3_phone_test.png", phone)
# cv2.imwrite("img4_remote.png", remote)