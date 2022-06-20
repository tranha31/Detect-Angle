from typing import final
import numpy as np
import cv2
import os

from bl import ProcessImg3_Add

class Image:
    def __init__(self, saveName, isDisturbance):
        self.saveName = saveName
        self.isDisturbance = isDisturbance

    def findObject(self, name, img):
        image = cv2.imread(name)
        oPIA = ProcessImg3_Add()
        finalImage = oPIA.findObject(image, self.isDisturbance)
        if name == "disturbance.png":
            os.remove(name)

        cv2.imwrite("phone.png", finalImage["phone"])
        cv2.imwrite("remote.png", finalImage["remote"])
        self.findAngle("phone.png", "remote.png", img)
    
    def findAngle(self, phoneName, remoteName, img):
        phone = cv2.imread(phoneName)
        remote = cv2.imread(remoteName)
        oPIA = ProcessImg3_Add()
        linePhone = oPIA.findPhoneAngle(phone)
        lineRemote = oPIA.findRemoteAngle(remote)
        os.remove(phoneName)
        os.remove(remoteName)

        x0 = (0, round(-linePhone['line'][2]/linePhone['line'][1]))
        x1 = (round(-linePhone['line'][2]/linePhone['line'][0]), 0)

        x2 = (0, round(-lineRemote['line'][2]/lineRemote['line'][1]))
        x3 = (round(-lineRemote['line'][2]/lineRemote['line'][0]), 0)
        
        cv2.line(img, x0, x1, (255,0,0), 2)
        cv2.line(img, x2, x3, (255,0,0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        a1,b1,c1 = linePhone['line']
        a2,b2,c2 = lineRemote['line']
        num = a1*b2 - b1*a2
        den = a1*a2 + b1*b2
        if den != 0:
            theta = abs(np.arctan(num/den))*180/np.pi
            cv2.putText(img, str(round(theta, 1)), (img.shape[0]//2, img.shape[1]//4), font, 0.8, (255,0,0), 2, 0)

        # cv2.imshow("angle", img)
        # cv2.waitKey(0)
        cv2.imwrite("findImage/" + self.saveName, img)

class Img3_Add(Image):
    
    def __init__(self, saveName, isDisturbance):
        super().__init__(saveName, isDisturbance)

    def detectAngleDisturbance(self):
        img = cv2.imread(self.saveName)
        oPIA = ProcessImg3_Add()
        image = oPIA.disturbanceHandler(img)
        cv2.imwrite("disturbance.png", image)
        self.findObject("disturbance.png", img)
    
    

class ImageOther(Image):
    def __init__(self, saveName, isDisturbance):
        super().__init__(saveName, isDisturbance)

    def detectAngle(self):
        img = cv2.imread(self.saveName)

        self.findObject(self.saveName, img)
    
    

img3Add = Img3_Add("img3_add.png", True)
# img3Add.detectAngleDisturbance()

img3_bruit = ImageOther("img3_bruit.png", False)
# img3_bruit.detectAngle()

img3_bruit2 = ImageOther("img3_bruit2.png", False)
# img3_bruit2.detectAngle()

img4 = ImageOther("img4.png", False)
img4.detectAngle()