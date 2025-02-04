import cv2
import numpy as np


def calculate_pitch_color_bounds(cap, num_samples=10):
    hsv_values = []

    # Sample frames from the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = total_frames // num_samples

    for i in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Assume the pitch is the dominant color in the frame
        hsv_values.append(hsv_frame.reshape(-1, 3))

    # Concatenate all sampled HSV values
    hsv_values = np.concatenate(hsv_values, axis=0)

    # Calculate the lower and upper bounds for the pitch color
    lower_bound = np.percentile(hsv_values, 15, axis=0)
    upper_bound = np.percentile(hsv_values, 95, axis=0)

    return lower_bound, upper_bound


input_video_path = 'E:/Downloads/Futball.2024.12.22.Monza.vs.Juventus.HDTV.1080i.Hun-BLG/raw_video.mkv'
output_video_path = 'E:/Videos/no_background_dynamic.mp4'

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps,
                      (frame_width, frame_height), isColor=True)

# Calculate the frame range for the desired timestamps
start_time = 4 * 60 + 46  # 2:12 in seconds
end_time = 8 * 60 + 46  # 10:00 in seconds
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# Calculate the pitch color bounds
lower_green, upper_green = calculate_pitch_color_bounds(cap)

# Set the starting frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

while True:
    ret, img = cap.read()
    print(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
        break  # Exit the loop if no more frames or end frame reached

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    pitch_only = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(pitch_only, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    contours, _ = cv2.findContours(
        blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [largest_contour], -1,
                         (255, 255, 255), thickness=cv2.FILLED)
        pitch_only = cv2.bitwise_and(img, mask)

    out.write(pitch_only)
    cv2.imshow('Pitch Only Video', pitch_only)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
