from mimetypes import init
import numpy as np
import cv2
import math

class ProcessImg3_Add:
    tLower = 70
    tUpper = 200
    blurSize = 7
    filt = np.ones((5, 5), dtype = int)

    def __init__(self) -> None:
        pass
    
    # Bộ lọc Sobel theo cả x và y
    # Để loại bỏ những đường // Ox và // Oy
    def sobelXY(self, img):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgGaussian = cv2.GaussianBlur(imgGray, (self.blurSize, self.blurSize), 0)
        sobelXY = cv2.Sobel(imgGaussian, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        return sobelXY

    # Phép làm tách rời các điểm ảnh
    def erosion(self, img, fil):
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

    # Phép nối các điểm ảnh
    def dilation(self, img, fil):
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

    # Phép đóng
    def closing(self, img, fil):
        c1 = self.dilation(img, fil)
        c2 = self.erosion(c1, fil)
        return c2

    # Phép mở
    def opening(self, img, fil):
        o1 = self.erosion(img, fil)
        o2 = self.dilation(o1, fil)
        return o2

    # Xử lý nhiễu với ảnh đầu vào
    # Nhiểu nặng: Có các đường sọc trên ảnh
    def disturbanceHandler(self, img):
        imageXY = self.sobelXY(img)
        edgexy = np.sqrt(imageXY**2)//30

        for i in range(0, len(edgexy)):
            for j in range(0, len(edgexy[i])):
                if edgexy[i][j] != 0:
                    edgexy[i][j] = 1
        closingImg = self.closing(edgexy, self.filt)
        openingImg = self.opening(closingImg, self.filt)
        finalImage = openingImg*255

        return finalImage
    
    # Tìm đường bao của đối tượng trong ảnh
    # Trả về điện thoại 
    def findPhone(self, image, imgThree, isDisturbance):
        contours, _ = cv2.findContours(imgThree,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        findContour = []
        for c in contours:
            area = cv2.contourArea(c)
            if isDisturbance == True:
                if area > 10000:
                    findContour.append(c)
            else:
                if area > 4000:
                    findContour.append(c)

        stencil = np.zeros(image.shape).astype(image.dtype)
        color = [255, 255, 255]

        phone = cv2.fillPoly(stencil, findContour[0], color)
        return phone

    # Tìm đường bao của đối tượng trong ảnh
    # Trả về điện thoại 
    def findRemote(self, image, imgThree, isDisturbance):
        contours, _ = cv2.findContours(imgThree,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        findContour = []
        for c in contours:
            area = cv2.contourArea(c)
            if isDisturbance == True:
                if area > 10000:
                    findContour.append(c)
            else:
                if area > 4000:
                    findContour.append(c)

        stencil = np.zeros(image.shape).astype(image.dtype)
        color = [255, 255, 255]
        index = 1
        if len(findContour) > 2:
            index = 2
        remote = cv2.fillPoly(stencil, findContour[index], color)
        return remote

    # Xác định đối tượng
    def findObject(self, img, isDisturbance):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imageBlur = cv2.GaussianBlur(imgGray, (self.blurSize, self.blurSize), 0)
        imageCanny = cv2.Canny(imageBlur, 60, 100)

        kernel = np.ones((5,5), dtype=int)
        imgDial = cv2.dilate(imageCanny, kernel=kernel, iterations=3)
        imgThree = cv2.erode(imgDial, kernel=kernel, iterations=2)

        phoneObject = self.findPhone(img, imgThree, isDisturbance)
        remoteObject = self.findRemote(img, imgThree, isDisturbance)
        filt = np.ones((11, 11), dtype = int)

        # Xác định điện thoại
        imgGray0 = cv2.cvtColor(phoneObject, cv2.COLOR_BGR2GRAY)
        imgGaussian = cv2.GaussianBlur(imgGray0, (self.blurSize, self.blurSize), 0)
        closingImg = self.closing(imgGaussian, filt)
        phone = closingImg*255

        # Xác định điều khiển
        imgGray1 = cv2.cvtColor(remoteObject, cv2.COLOR_BGR2GRAY)
        imgGaussian1 = cv2.GaussianBlur(imgGray1, (self.blurSize, self.blurSize), 0)
        closingImg1 = self.closing(imgGaussian1, filt)
        remote = closingImg1*255

        oObject = {
            "phone": phone,
            "remote": remote
        }
        return oObject

    # Xác định cạnh của điện thoại
    def findPhoneAngle(self, img):
        equations = self.findLine(img)
        result = list(filter(lambda x: math.atan(-x['line'][0]/x['line'][1])*(180/np.pi) <= 0, equations))
        line = min(result, key=lambda x:x['rho'])
        return line
    
    # Xác định cạnh của điều khiển
    def findRemoteAngle(self, img):
        equations = self.findLine(img)
        result = list(filter(lambda x: math.atan(-x['line'][0]/x['line'][1])*(180/np.pi) <= 0, equations))
        line = max(result, key=lambda x:x['rho'])
        return line
    
    # Xác định đường thẳng trên ảnh
    def findLine(self, image):
        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageBlur = cv2.GaussianBlur(imgGray, (self.blurSize, self.blurSize), 0)
        imageCanny = cv2.Canny(imageBlur, 60, 100)

        lines = cv2.HoughLinesP(imageCanny,1,np.pi/180,100,190,10)
        equations = []
        for line in lines:
            x1,y1,x2,y2 = line[0]
            rho = line[0][0]
            theta = line[0][1]
            lineFunc = np.cross([x1,y1,1],[x2,y2,1])

            equations.append(
                {'rho': rho, 'theta': theta, 'line': lineFunc, 'd1': (x1, y1), 'd2': (x2, y2)}
            )
        
        return equations
