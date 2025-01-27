import cv2
from cv2 import boundingRect
import numpy as np

lower_green = np.array([0, 0, 0])
upper_green = np.array([80, 255, 255])

input_video_path = 'E:/Downloads/Futball.2024.12.22.Monza.vs.Juventus.HDTV.1080i.Hun-BLG/raw_video.mkv'

# Path to the output video file
output_video_path = 'Videos/no_background.mp4'

# Open the video capturer
cap = cv2.VideoCapture(input_video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video frame width, height, and frames per second (fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps,
                      (frame_width, frame_height), isColor=True)

frameCountLimit = 5000
frameIndex = 0
while True:
    ret, img = cap.read()
    orig_img = img

    if not ret:
        break  # Exit the loop if no more frames

    # Convert the frame to grayscale
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    mask = np.zeros_like(orig_img)
    cv2.drawContours(mask, [largest_contour], -1,
                     (255, 255, 255), thickness=cv2.FILLED)
    roi = cv2.bitwise_and(orig_img, mask)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the region of interest
    cropped_roi = roi[y:y+h, x:x+w]

    # Create a blank image with the same size as the cropped ROI
    blank_image = np.zeros_like(orig_img)

# Paste the ROI onto the blank image
    blank_image = cv2.add(blank_image, roi)

    # Write the grayscale frame to the output video file
    out.write(blank_image)

    # Display the frame (optional)
    cv2.imshow('Grayscale Video', blank_image)

    # Press 'q' to exit the display window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frameIndex += 1

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
