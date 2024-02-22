import cv2
from cv2 import boundingRect
from matplotlib.lines import Line2D
from sympy import Point, Line, pi
import math
import numpy as np
import random as rng
from shapely.geometry import LineString

rng.seed(12345)


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def angle(self):
        return angle_of_vectors(Line(x1, y1, x2, y2), Line(0, 0, 1, 0))


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def angle_of_vectors(line1, line2):

    a = line1.x2-line1.x1
    b = line1.y2-line1.y1
    c = line2.x2-line2.x1
    d = line2.y2-line2.y1

    dotProduct = a*c + b*d
    # for three dimensional simply add dotProduct = a*c + b*d  + e*f
    modOfVector1 = math.sqrt(a*a + b*b)*math.sqrt(c*c + d*d)
    # for three dimensional simply add modOfVector = math.sqrt( a*a + b*b + e*e)*math.sqrt(c*c + d*d +f*f)
    angle = dotProduct/modOfVector1
    if angle > 1:
        angle = 1.0

    rad = math.acos(angle)
    # print("Cosθ =",angle)
    angleInDegree = math.degrees(rad)
    return angleInDegree


def distanceBetLine(line1, line2):
    points1 = []
    points2 = []
    points1.append(Point(line1.x1, line1.y1))
    points1.append(Point(line1.x2, line1.y2))
    points2.append(Point(line2.x1, line2.y1))
    points2.append(Point(line2.x2, line2.y2))
    minDistance = 1000000000
    for item in points1:
        for item2 in points2:
            result = ((((item2.x - item.x)**2) + ((item2.y-item.y)**2))**0.5)
            if (result < minDistance):
                minDistance = result
    return minDistance


img = cv2.imread("center.jpg")


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold of blue in HSV space
lower_blue = np.array([26, 20, 10])
upper_blue = np.array([86, 255, 255])

# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
result = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow('frame', img)
# cv2.imshow('mask', mask)
# cv2.imshow('result', result)


img2 = result
img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
# sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
# sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

# cv2.imshow('Sobel Y', sobely)
# cv2.imshow('Sobel X', sobelx)
# cv2.imshow('Sobel XY', sobelxy)

# cv2.imshow("image", img_blur)
# cv2.imshow("image_gray",img_gray )

contours, hierarchy = cv2.findContours(
    img_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
centers = [None]*len(contours)
radius = [None]*len(contours)
maxIndex = 0
maxValue = 0
for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    # print(contours_poly[i].size)
    boundRect[i] = cv2.boundingRect(contours_poly[i])
    area = boundRect[i][2]*boundRect[i][3]
    if (area > maxValue):
        maxValue = area
        maxIndex = i
    # print(area)
    centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

mask2 = np.zeros(img.shape[:2], dtype=np.uint8)
# for i in range(len(contours)):
#    if(contours[i].size==304):
# print(contours[i].size)
#        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
cv2.drawContours(mask2, contours_poly, maxIndex, (255, 255, 255), -1)
# cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
#  (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
# cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)


# cv2.imshow('Contours', img_gray)
masked = cv2.bitwise_and(img_gray, img_gray, mask=mask2)
mask3 = np.zeros(img.shape[:2], dtype=np.uint8)
# cv2.imshow('Masked',masked)
# masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
img_blur = masked  # cv2.GaussianBlur(masked, (3,3), 0)
# cv2.imshow("Mask Applied to Image", img_blur)
edges = cv2.Canny(image=img_blur, threshold1=80,
                  threshold2=220)  # Canny Edge Detection
cv2.drawContours(edges, contours_poly, maxIndex, (0, 0, 0), 3)
lines = cv2.HoughLines(edges, 0.02, np.pi/360, threshold=10)
# cv2.imshow("Edges", edges)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imshow("Edges", mask3)
contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
centers = [None]*len(contours)
radius = [None]*len(contours)
maxIndex = 0
maxValue = 0
for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    # print(contours_poly[i].size)
    boundRect[i] = cv2.boundingRect(contours_poly[i])
    area = boundRect[i][2]*boundRect[i][3]
    if (area > maxValue):
        maxValue = area
        maxIndex = i
    # print(area)
    centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

mask3 = np.zeros(img.shape[:2], dtype=np.uint8)
for i in range(len(contours)):
    if (contours[i].size > 0):
        # print(contours[i].size)
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(mask3, contours_poly, maxIndex, (255, 255, 255), -1)
        cv2.rectangle(mask3, (int(boundRect[i][0]), int(boundRect[i][1])),
                      (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv2.circle(mask3, (int(centers[i][0]), int(
            centers[i][1])), int(radius[i]), color, 2)

cv2.imshow('canny', edges)
cv2.imshow('image', img)
cv2.waitKey(0)
