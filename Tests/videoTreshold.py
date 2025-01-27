import cv2
import numpy as np

# Open a video file or capture device
# Replace 'video_path.mp4' with 0 for webcam
video_capture = cv2.VideoCapture('Videos/no_background.mp4')

if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read the frame from the video
    ret, frame = video_capture.read()

    # Break the loop if no frame is returned
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the desired color (e.g., black color)
    lower_bound = np.array([0, 0, 200])
    upper_bound = np.array([255, 50, 255])

    # Apply the threshold
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Optional: Mask the original frame to show only the desired color
    result_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original frame, mask, and result
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
