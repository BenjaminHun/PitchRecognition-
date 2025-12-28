import cv2
import numpy as np
import os


def calculate_dominant_hue(image, mask=None):
    """Calculate the dominant hue in the image, optionally using a mask."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel = hsv_image[:, :, 0]

    if mask is not None:
        h_channel = h_channel[mask > 0]

    hist, _ = np.histogram(h_channel, bins=180, range=(0, 180))
    dominant_hue = np.argmax(hist)
    return dominant_hue


def segment_foreground_by_hue(image, hue_tolerance=10):
    """Segment the foreground using hue-based background removal."""
    # Assume the background is the border of the image
    border_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    border_thickness = 10  # Pixels from the border to consider as background
    border_mask[:border_thickness, :] = 255
    border_mask[-border_thickness:, :] = 255
    border_mask[:, :border_thickness] = 255
    border_mask[:, -border_thickness:] = 255

    # Calculate the dominant hue of the background
    dominant_hue = calculate_dominant_hue(image, mask=border_mask)

    # Define the hue range for background removal
    lower_bound = np.array([max(0, dominant_hue - hue_tolerance), 50, 50])
    upper_bound = np.array([min(179, dominant_hue + hue_tolerance), 255, 255])

    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the background
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Invert the mask to keep the foreground
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=mask_inv)

    return result


def remove_background_by_auto_hue(input_folder, output_folder, hue_tolerance=10):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Unable to load image {filename}")
                continue

            # Assume the background is the border of the image
            border_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            border_thickness = 10  # Pixels from the border to consider as background
            border_mask[:border_thickness, :] = 255
            border_mask[-border_thickness:, :] = 255
            border_mask[:, :border_thickness] = 255
            border_mask[:, -border_thickness:] = 255

            # Calculate the dominant hue of the background
            dominant_hue = calculate_dominant_hue(image, mask=border_mask)
            #print(f"Dominant hue for {filename}: {dominant_hue}")

            # Define the hue range for background removal
            lower_bound = np.array(
                [max(0, dominant_hue - hue_tolerance), 50, 50])
            upper_bound = np.array(
                [min(179, dominant_hue + hue_tolerance), 255, 255])

            # Convert the image to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Create a mask for the background
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

            # Invert the mask to keep the foreground
            mask_inv = cv2.bitwise_not(mask)

            # Apply the mask to the image
            result = cv2.bitwise_and(image, image, mask=mask_inv)

            # Save the result
            cv2.imwrite(output_path, result)
            #print(f"Processed and saved: {output_path}")


# Input and output directories
input_folder = 'E:/raw_segmentation'
output_folder = 'E:/removed_background'

# Call the function with automatic background removal
remove_background_by_auto_hue(input_folder, output_folder, hue_tolerance=10)
