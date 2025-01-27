from PIL import Image
import os

import cv2
import numpy as np

# Specify the directory path
directory_path = 'E:/numberTest/'
testImage = "41_39.jpg"


def nothing(x):
    pass


# Load an image
image = cv2.imread(directory_path+testImage)
scale_x = 4.0  # Scale factor along the horizontal axis
scale_y = 4.0  # Scale factor along the vertical axis

# Upsample the image using the scaling factors
image = cv2.resize(
    image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Create a window
cv2.namedWindow('image')

# Create trackbars for hue, saturation, and value ranges
cv2.createTrackbar('H_min', 'image', 0, 179, nothing)
cv2.createTrackbar('H_max', 'image', 179, 179, nothing)
cv2.createTrackbar('S_min', 'image', 0, 255, nothing)
cv2.createTrackbar('S_max', 'image', 255, 255, nothing)
cv2.createTrackbar('V_min', 'image', 0, 255, nothing)
cv2.createTrackbar('V_max', 'image', 255, 255, nothing)

while True:
    # Get current positions of trackbars
    h_min = cv2.getTrackbarPos('H_min', 'image')
    h_max = cv2.getTrackbarPos('H_max', 'image')
    s_min = cv2.getTrackbarPos('S_min', 'image')
    s_max = cv2.getTrackbarPos('S_max', 'image')
    v_min = cv2.getTrackbarPos('V_min', 'image')
    v_max = cv2.getTrackbarPos('V_max', 'image')

    # Set the lower and upper HSV range according to the values selected by the trackbars
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    # Create a mask based on the HSV range
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Display the original image, the mask, and the result side by side
    combined = np.hstack(
        (image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result))
    cv2.imshow('image', combined)

    # Break the loop when the user hits the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the window
cv2.destroyAllWindows()
