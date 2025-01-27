import cv2
import matplotlib.pyplot as plt

# Step 1: Read the image


def roi():
    directory_path = 'E:/numberTest/'
    testImage = "41_39.jpg"
    image = cv2.imread(directory_path+testImage)

# Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Get the dimensions of the image
    height, width, _ = image.shape

# Step 3: Calculate the middle third coordinates
    start_row = height // 4
    end_row = start_row * 2

# Step 4: Extract the middle third of the image
    middle_third = image_rgb[start_row:end_row, :]
    return image_rgb, middle_third


image_rgb, middle_third = roi()

# Display the original and the middle third image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(middle_third)
plt.title('Middle Third of the Image')

plt.show()
