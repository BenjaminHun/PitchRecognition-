import copy
from PIL import Image
import os
import pytesseract
import cv2
import numpy as np
from imutils import paths
import argparse
import cv2
import re
from PIL import Image

# Specify the directory path
directory_path = 'E:/test/'
testImage = "21_14.jpg"
globalLabeledImagesIndex = 0
imshowIndex = 0


contourSizeHeightMin = 35
contourSizeHeightMax = 70
contourSizeWidthMin = 20
contourSizeWidthMax = 90

contourAreaMin = 400
contourAreaMax = 2000

proportion = 10
start_row = 2
end_row = 5

blurTresholdMin = 3500
blurTresholdMax = 4000


def extract_numbers(string):
    numbers = re.findall(r'\d+', string)
    return list(map(int, numbers))


def stack_images_cv2(image1, image2, output_path, direction='horizontal'):
    # Resize images to the same width or height depending on the stacking direction
    if direction == 'vertical':
        # Match the width of the images
        new_width = min(image1.shape[1], image2.shape[1])
        image1 = cv2.resize(image1, (new_width, int(
            image1.shape[0] * (new_width / image1.shape[1]))))
        image2 = cv2.resize(image2, (new_width, int(
            image2.shape[0] * (new_width / image2.shape[1]))))

        # Stack images vertically
        stacked_image = np.vstack((image1, image2))

    elif direction == 'horizontal':
        # Match the height of the images
        new_height = min(image1.shape[0], image2.shape[0])
        image1 = cv2.resize(
            image1, (int(image1.shape[1] * (new_height / image1.shape[0])), new_height))
        image2 = cv2.resize(
            image2, (int(image2.shape[1] * (new_height / image2.shape[0])), new_height))

        # Stack images horizontally
        stacked_image = np.hstack((image1, image2))

    # Save the new image
    cv2.imwrite(output_path, stacked_image)


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()


def prepareImage(image):

    scale_x = 4.0  # Scale factor along the horizontal axis
    scale_y = 4.0  # Scale factor along the vertical axis

    image = cv2.resize(
        image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("roi", image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.namedWindow('image')

    global team1Lower_hsv
    global team1Upper_hsv

    global team2Lower_hsv
    global team2Upper_hsv

    team1Mask = cv2.bitwise_not(cv2.inRange(
        image_hsv, team1Lower_hsv, team1Upper_hsv))
    team2Mask = cv2.bitwise_not(cv2.inRange(
        image_hsv, team2Lower_hsv, team2Upper_hsv))

    return image, team1Mask, team2Mask


def getContourImages(image, mask):
    global imshowIndex
    global contourSizeHeightMin
    global contourSizeHeightMax
    global contourSizeWidthMin
    global contourSizeWidthMax
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_images = []
    for contour in contours:
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Crop the image using the bounding box
        if contourSizeHeightMin < h < contourSizeHeightMax and contourSizeWidthMin < w < contourSizeWidthMax:
            contour_image = mask[y:y+h, x:x+w]
            # cv2.imshow(str(imshowIndex), contour_image)
            imshowIndex += 1
            # Append the cropped image to the list
            contour_images.append((contour_image, cv2.contourArea(contour)))
    return contours, contour_images


def recognizeContourNumbers(recognizedNumbers, contour_images, fileName, originalImg):
    global contourAreaMin
    global contourAreaMax
    isContainedNumber = False

    for i, img in enumerate(contour_images):
        cv2.imshow("numbers", img[0])
        if not contourAreaMin < img[1] < contourAreaMax:
            continue
        data = pytesseract.image_to_data(
            img[0], config='--psm 8 -c tessedit_char_whitelist=0123456789', output_type=pytesseract.Output.DICT)
        text = ""
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 80:  # Only consider high-confidence results
                text += data['text'][i]

        if text != "":
            global globalLabeledImagesIndex
            isContainedNumber = True
            recognizedNumbers.append(
                str(globalLabeledImagesIndex)+"|"+str(fileName)+":"+text)

            globalLabeledImagesIndex += 1
            # cv2.imwrite("E:\\labeledImages\\" +
            #            str(globalLabeledImagesIndex)+"_"+str(text)+".jpg", img[0])
            stack_images_cv2(originalImg, np.stack((img[0],) * 3, axis=-1), "E:\\labeledImages\\" +
                             str(globalLabeledImagesIndex)+"_"+str(text)+".jpg")
        print(text+" "+str(img[1]))
    return isContainedNumber


def roi(directory_path, file):
    global proportion
    global start_row
    global end_row
    global imshowIndex
    # directory_path = 'E:/numberTest/'
    # testImage = "41_39.jpg"
    image = cv2.imread(directory_path+file)

# Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Get the dimensions of the image
    height, width, _ = image.shape

# Step 3: Calculate the middle third coordinates

# Step 4: Extract the middle third of the image
    middle_third = image_rgb[(int)(height/proportion) *
                             start_row:(int)(height/proportion)*end_row, :]

    return middle_third


def filterFilesByIndex(i, all_files, filtered_files):
    for file in all_files:
        if file.startswith(str(i)+"_"):
            filtered_files.append(file)


for i in range(400):
    # i = 14
    all_files = os.listdir(directory_path)
    filtered_files = []
    recognizedNumbers = []
    maximumRecognizedImage = 10
    maximumRecognizedImageIndex = 0
    filterFilesByIndex(i, all_files, filtered_files)
    filtered_files = sorted(filtered_files, key=extract_numbers)
    for file in filtered_files:
        img = cv2.imread(directory_path+file)
        # Get the dimensions of the image
        height, width, channels = img.shape
        if width < 50:
            continue
        if file == "3_2.jpg":
            a = 5
        print(file)
        # a középső régió ahol a szám található
        blurness = variance_of_laplacian(img)
        print("blur: "+(str)(blurness))
        if not blurTresholdMin < blurness < blurTresholdMax:
            continue
        else:
            a = 21
        image = roi(directory_path, file)

        # a mezek színe szerint maszkolunk csapatonként
        image, team1Mask, team2Mask = prepareImage(
            image)
        concat_images = np.hstack((team1Mask, team2Mask))
        cv2.imshow("masks", concat_images)

        # meghatározott méretu és méretarányu konturokat kinyerjuk
        team1Contours, team1ContourImages = getContourImages(
            image, team1Mask)

        team2Contours, team2ContourImages = getContourImages(
            image, team2Mask)
        # ha a konturok valamelyikén van mezszám akkor növeljük az indexet
        if (recognizeContourNumbers(recognizedNumbers, team1ContourImages, file, image) or
                recognizeContourNumbers(recognizedNumbers, team2ContourImages, file, image)):
            maximumRecognizedImageIndex += 1

        if maximumRecognizedImageIndex > maximumRecognizedImage:
            break
        # Break the loop when the user hits the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if len(recognizedNumbers) > 0:
        with open('recognizedNumbers.txt', 'a') as fp:
            fp.write('\n'.join(recognizedNumbers))
            fp.write('\n')
cv2.waitKey(0)
cv2.destroyAllWindows()
