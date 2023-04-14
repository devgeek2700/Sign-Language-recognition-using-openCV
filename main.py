import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300

folder = "Images_1/More"
count = 0

while True:
    success,img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand["bbox"]

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        cv2.imshow("IMG_WHITE", imgWhite)

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


        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap,:] = imgResize


        cv2.imshow("IMG_CROP", imgCrop)
        cv2.imshow("IMG_WHITE", imgWhite)

    cv2.imshow("LIVE_VIDEO", img)
    key = cv2.waitKey(1)
    if key == ord('a'):
        count = count + 1;
        cv2.imwrite(f'{folder}/Images_{time.time()}.jpg', imgWhite)
        print(count)







