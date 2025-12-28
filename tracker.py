import cv2
from Classes.TrackedObject import TrackedObject
from Classes.CurrentTrackedObject import CurrentTrackedObject
from utils import get_centroid

class TrackingManager:
    def __init__(self):
        self.trackedObjects = []
        self.id = 0
        self.whUpperRatio = 5
        self.whBottomRatio = 1.3

    def track_objects(self, contours):
        """Create CurrentTrackedObject instances for valid contours.
        Args:
            contours: List of contours to track
        Returns:
            List of CurrentTrackedObject instances
        """
        tracks = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            centroid = get_centroid((x, y, w, h))
            ratio = h / w
            if self.whBottomRatio < ratio < self.whUpperRatio:
                tracks.append(CurrentTrackedObject(
                    (x, y, w, h), contour, centroid))
        return tracks

    def update_tracked_object(self, originalFrame, currentTrackedObject, trackedObject):
        """Update tracked object with new contour and position."""
        trackedObject.addContour(currentTrackedObject.contour)
        trackedObject.actualizeCenterPos(currentTrackedObject.centerPos)
        trackedObject.actualizeRectangle(currentTrackedObject.rectangle)
        trackedObject.actualizeFrame(originalFrame)
        trackedObject.currentPairFound = True
        currentTrackedObject.connected = True

    def finalize_tracked_objects(self):
        """Remove dead objects and finalize alive ones."""
        for trackedObject in self.trackedObjects:
            trackedObject.final()
            if not trackedObject.isAlive:
                trackedObject.print_max_area_difference()
        
        self.trackedObjects[:] = [
            obj for obj in self.trackedObjects if obj.isAlive]

    def fill_empty_tracked_objects(self, originalFrame, currentTrackedObjects):
        """Initialize tracked objects list if empty."""
        for item in currentTrackedObjects:
            self.trackedObjects.append(TrackedObject(
                item.rectangle, item.contour, item.centerPos, originalFrame, self.id))
            self.id += 1

    def add_new_tracked_objects(self, originalFrame, currentTrackedObjects):
        """Add new objects that were not matched to existing tracks."""
        for currentTrackedObject in currentTrackedObjects:
            if not currentTrackedObject.connected:
                self.trackedObjects.append(TrackedObject(
                    currentTrackedObject.rectangle, currentTrackedObject.contour,
                    currentTrackedObject.centerPos, originalFrame, self.id))
                self.id += 1

    def process_frame(self, originalFrame, contours):
        """Process one frame of tracking.
        Args:
            originalFrame: The original input frame
            contours: List of detected contours
        Returns:
            List of current tracked objects
        """
        currentTrackedObjects = self.track_objects(contours)
        
        if len(self.trackedObjects) == 0:
            self.fill_empty_tracked_objects(originalFrame, currentTrackedObjects)
            return self.trackedObjects

        # Try to match current objects to existing tracks
        for currentTrackedObject in currentTrackedObjects:
            if currentTrackedObject.connected:
                continue
            for trackedObject in self.trackedObjects:
                if not trackedObject.isAlive or trackedObject.currentPairFound:
                    continue
                result = trackedObject.calculateDistance(
                    currentTrackedObject.centerPos)
                if result != -1:
                    self.update_tracked_object(
                        originalFrame, currentTrackedObject, trackedObject)

        self.finalize_tracked_objects()
        self.add_new_tracked_objects(originalFrame, currentTrackedObjects)
        
        return self.trackedObjects