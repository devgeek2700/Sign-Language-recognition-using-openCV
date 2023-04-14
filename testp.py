
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

classifier = Classifier("Model/keras_modelnag.h5","Model/labelsnag.txt")

offset = 20
imgSize = 300


folder = "Images/M"
count = 0

labelsnag = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
          'P','Q','R','S','T','U','V','W','X','Y',"Don't Want","Hello","Help","I Love You","Money","No","Ok",
          "Thank You","Want","Yes"]


while True:
    success,img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand["bbox"]

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255

        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        if aspectRatio > 1:
            k=imgSize/h
            wCal= math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            widGap= math.ceil((imgSize-wCal)/2)
            imgWhite[:, widGap:wCal + widGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)


        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)

        # cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,1.8,(255,0,250),2)
        cv2.putText(imgOutput, labelsnag[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1.8, (255, 0, 250), 2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),3)


    cv2.imshow(" SIGN LANGAUGE RECOGNITION.... ", imgOutput)
    if cv2.waitKey(1)==13:
        break
