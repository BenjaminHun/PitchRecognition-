import cv2
import numpy as np

def prepare_image(frame, canny_threshold1=40, canny_threshold2=200):
    """Convert frame to grayscale, apply Canny edge detection and adaptive threshold.
    Args:
        frame: Input BGR image
        canny_threshold1: First threshold for Canny edge detector
        canny_threshold2: Second threshold for Canny edge detector
    Returns:
        Processed binary image
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, canny_threshold1, canny_threshold2)
    edges = cv2.adaptiveThreshold(
        edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return edges

def filter_contours(contours, area_min=1500, area_max=5000):
    """Filter contours by area thresholds.
    Args:
        contours: List of contours to filter
        area_min: Minimum contour area
        area_max: Maximum contour area
    Returns:
        List of filtered contours
    """
    return [
        contour for contour in contours
        if area_min < cv2.contourArea(contour) < area_max
    ]

def draw_tracks(image, tracks):
    """Draw rectangles and IDs for all alive tracks.
    Args:
        image: Input image (BGR or grayscale)
        tracks: List of tracked objects
    Returns:
        Image with visualized tracks
    """
    # Convert image to BGR if needed
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # If image is RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
    # Draw each track
    for track in tracks:
        if not track.isAlive:
            continue
        x, y, w, h = track.rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(image, track.centerPos, 5, (0, 0, 255), -1)
        cv2.putText(image, str(track.id), track.centerPos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 255))
    return image