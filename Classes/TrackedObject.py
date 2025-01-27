import math

import cv2
import numpy as np


class TrackedObject:
    def __init__(self, rectangle, contour, centerPos, frame, id) -> None:
        self.startTtl = 10
        self.ttl = self.startTtl
        self.contourHistory = []
        self.isAlive = True
        self.distanceLimit = 40
        self.contour = contour
        self.centerPos = centerPos
        self.currentPairFound = False
        self.rectangle = rectangle
        self.frame = frame
        self.id = id

    def calculateDistance(self, centerPos):
        dist = math.dist(self.centerPos, centerPos)
        if dist < self.distanceLimit:
            return dist
        else:
            return -1

    def actualizeCenterPos(self, centerPos):
        self.centerPos = centerPos

    def actualizeRectangle(self, rectangle):
        self.rectangle = rectangle

    def addContour(self, contour):
        self.contourHistory.append(contour)
        self.currentPairFound = True

    def actualizeFrame(self, frame):
        self.frame = frame

    def reduceTtl(self):
        self.ttl -= 1
        if self.ttl == 0:
            self.isAlive = False

    def restoreTtl(self):
        self.ttl = self.startTtl

    def final(self):
        if not self.isAlive:
            return
        if not self.currentPairFound:
            self.reduceTtl()
        else:
            self.restoreTtl()
            self.currentPairFound = False
            self.saveContours()

    def saveContours(self):
        index = len(self.contourHistory)
        x, y, w, h = self.rectangle
        mask = np.zeros((h, w), dtype=np.uint8)
        # Shift the contour to the top-left of the bounding rectangle
        contour_shifted = self.contourHistory[index-1] - [x, y]
        cv2.drawContours(mask, [contour_shifted], -1,
                         (255), thickness=cv2.FILLED)

        # Crop the original image to the bounding rectangle
        cropped_image = self.frame[y:y+h, x:x+w]

        # Apply the mask to the cropped image
        result = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
        a = 1
        # cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.circle(self.frame, self.centerPos, 5, (0, 0, 255), -1)
        # cv2.putText(self.frame, str(self.id), self.centerPos,
        #           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 255))
        # cv2.imshow('Edge Detected Video', self.frame)

        # To save the result as an image
        cv2.imwrite(str("E:/test/"+str(self.id)+"_"+str(index)+".jpg"), result)
