import math

class CurrentTrackedObject:
    def __init__(self,rectangle, contour, centerPos) -> None:
        self.contour = contour
        self.centerPos = centerPos
        self.connected = False
        self.rectangle=rectangle