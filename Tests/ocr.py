import cv2
import numpy as np
import pytesseract
from PIL import Image

import os

# Specify the directory path
directory_path = 'E:/numberTest/'
testImage = "41_39.jpg"

# Get a list of all files and directories
all_files = os.listdir(directory_path)
# Load the image

img = cv2.imread(directory_path+testImage)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scale_x = 4.0  # Scale factor along the horizontal axis
scale_y = 4.0  # Scale factor along the vertical axis

# Upsample the image using the scaling factors
img = cv2.resize(
    img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
cv2.imshow("gray", img)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
img = cv2.filter2D(img, -1, kernel)
cv2.imshow("enhanced", img)
img = cv2.Canny(img, 100, 210)
cv2.imshow("edge", img)

# Perform OCR with confidence scores
data = pytesseract.image_to_data(
    img, config='--psm 8 -c tessedit_char_whitelist=0123456789', output_type=pytesseract.Output.DICT)

# Extract text and confidence scores
text = ""
n_boxes = len(data['level'])
for i in range(n_boxes):
    if int(data['conf'][i]) > 80:  # Only consider high-confidence results
        text += data['text'][i]

if text != "":
    print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()
