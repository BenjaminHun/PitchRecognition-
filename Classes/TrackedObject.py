import math
import cv2
import numpy as np
from segmentation import segment_foreground_by_hue


class TrackedObject:
    def __init__(self, rectangle, contour, centerPos, frame, id) -> None:
        # Tracking parameters
        self.startTtl = 10
        self.ttl = self.startTtl
        self.distanceLimit = 40

        # Object state
        self.contourHistory = []
        self.isAlive = True
        self.contour = contour
        self.centerPos = centerPos
        self.currentPairFound = False
        self.rectangle = rectangle
        self.frame = frame
        self.id = id

        # Initialize Kalman filter: 4 states (x, y, dx, dy), 2 measurements (x, y)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.array(
            [centerPos[0], centerPos[1], 0, 0], np.float32)

        # Area tracking attributes
        self.prev_area = None
        self.max_area_diff = 0

    def predict(self):
        """Predict the next position of the object using the Kalman filter."""
        prediction = self.kalman.predict()
        return int(prediction[0]), int(prediction[1])

    def update(self, centerPos):
        """Update the Kalman filter with the new measurement and set the new center position."""
        measurement = np.array([centerPos[0], centerPos[1]], np.float32)
        self.kalman.correct(measurement)
        self.centerPos = (int(self.kalman.statePost[0]), int(
            self.kalman.statePost[1]))

    def calculateDistance(self, centerPos):
        """Calculate the distance between the current position and a new position.
        Returns the distance if within limit, otherwise -1."""
        dist = math.dist(self.centerPos, centerPos)
        return dist if dist < self.distanceLimit else -1

    def actualizeCenterPos(self, centerPos):
        """Update the center position using the Kalman filter."""
        self.update(centerPos)

    def actualizeRectangle(self, rectangle):
        """Update the bounding rectangle."""
        self.rectangle = rectangle

    def addContour(self, contour):
        """Add a new contour to the history and mark as paired this frame."""
        self.contourHistory.append(contour)
        self.currentPairFound = True

    def actualizeFrame(self, frame):
        """Update the stored frame for this object."""
        self.frame = frame

    def reduceTtl(self):
        """Decrease TTL; mark as dead if TTL reaches zero."""
        self.ttl -= 1
        if self.ttl == 0:
            self.isAlive = False

    def restoreTtl(self):
        """Restore TTL to its initial value."""
        self.ttl = self.startTtl

    def final(self):
        """Finalize the object for the current frame.
        Reduce TTL if not paired, otherwise restore TTL and save contours."""
        if not self.isAlive:
            return
        if not self.currentPairFound:
            self.reduceTtl()
        else:
            self.restoreTtl()
            self.currentPairFound = False
            self.saveContours()

    def saveContours(self):
        """Save the segmented foreground of the object as an image."""
        index = len(self.contourHistory)
        x, y, w, h = self.rectangle

        # Crop the frame to the bounding rectangle
        cropped_frame = self.frame[y:y + h, x:x + w]

        # Segment the foreground using hue
        result = segment_foreground_by_hue(cropped_frame, hue_tolerance=10)

        # Save the result as an image
        cv2.imwrite(str("E:/removed_background/" + str(self.id) +
                    "_" + str(index) + ".jpg"), result)

    def print_max_area_difference(self):
        if not hasattr(self, 'contourHistory'):
            print(f"Object {self.id}: No contourHistory attribute")
            return
        
        print(f"Object {self.id}: Number of contours in history: {len(self.contourHistory)}")
        
        if len(self.contourHistory) < 2:
            print(f"Object {self.id}: Not enough contours to calculate area difference (need at least 2)")
            return

        areas = [cv2.contourArea(contour) for contour in self.contourHistory]
        max_diff = max(abs(areas[i] - areas[i-1])
                       for i in range(1, len(areas)))
        print(f"Maximum area difference for object {self.id}: {max_diff}")
