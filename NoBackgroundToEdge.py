import cv2
from Classes.TrackedObject import TrackedObject
from Classes.CurrentTrackedObject import CurrentTrackedObject
from copy import copy

whUpperRatio = 5
whBottomRatio = 1.3

area_threshold1 = 1500
area_threshold2 = 5000

cannyEdgeTreshold1 = 40
cannyEdgeTreshold2 = 200

blurKernel = 9


# Path to the input video file
input_video_path = 'Videos/no_background.mp4'

# Path to the output video file
output_video_path = 'Videos/edge_detected.mp4'

# Open the video capture
cap = cv2.VideoCapture(input_video_path)
trackedObjects = []
id = 0
# Check if the video opened successfully


def setVideoParameters(output_video_path, cap):
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
                          (frame_width, frame_height), isColor=False)

    return out


def get_centroid(bbox):
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    return (cx, cy)


def draw_tracks(image, tracks):
    image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for track in tracks:
        if not track.isAlive:
            continue
        x, y, w, h = track.rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(image, track.centerPos, 5, (0, 0, 255), -1)
        cv2.putText(image, str(track.id), track.centerPos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 255))
    return image


def track_objects(contours):
    tracks = []
    global whUpperRatio
    global whBottomRatio
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        centroid = get_centroid((x, y, w, h))
        ratio = h/w
        if whBottomRatio < ratio < whUpperRatio:
            tracks.append(CurrentTrackedObject(
                (x, y, w, h), contour, centroid))
    return tracks


def filterContours(contours):
    global area_threshold1
    global area_threshold2
    trackedContours = []
    for contour in contours:
        if cv2.contourArea(contour) > area_threshold1 and cv2.contourArea(contour) < area_threshold2:
            trackedContours.append(contour)

    return trackedContours


def prepareImage(frame):
    global cannyEdgeTreshold1
    global cannyEdgeTreshold2
    global blurKernel
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_frame, cannyEdgeTreshold1, cannyEdgeTreshold2)
    _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.GaussianBlur(edges, (blurKernel, blurKernel), 0)
    return edges


out = setVideoParameters(output_video_path, cap)
# Process the video frame by frame


def updateTrackedObject(originalFrame, currentTrackedObject, trackedObject):
    trackedObject.addContour(currentTrackedObject.contour)
    trackedObject.actualizeCenterPos(
        currentTrackedObject.centerPos)
    trackedObject.actualizeRectangle(
        currentTrackedObject.rectangle)
    trackedObject.actualizeFrame(originalFrame)
    trackedObject.currentPairFound = True
    currentTrackedObject.connected = True


def finalizeTrackedObjects(trackedObjects):
    for trackedObject in trackedObjects:
        trackedObject.final()


def fillEmptyTrackedObjects(trackedObjects, originalFrame, currentTrackedObjects):
    global id
    for item in currentTrackedObjects:
        trackedObjects.append(TrackedObject(
            item.rectangle, item.contour, item.centerPos, originalFrame, id))
        id += 1


def addNewTrackedObjects(trackedObjects, originalFrame, currentTrackedObjects):
    global id
    for currentTrackedObject in currentTrackedObjects:
        if not currentTrackedObject.connected:
            trackedObjects.append(TrackedObject(
                currentTrackedObject.rectangle, currentTrackedObject.contour, currentTrackedObject.centerPos, originalFrame, id))
            id += 1


while True:
    ret, frame = cap.read()
    originalFrame = copy(frame)

    if not ret:
        break  # Exit the loop if no more frames

    frame = prepareImage(frame)
    contours, hierarchy = cv2.findContours(
        frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = filterContours(contours)
    cv2.drawContours(frame, contours, -1,
                     (122, 255, 122), thickness=cv2.FILLED)

    # Write the edge-detected frame to the output video file
    # out.write(edges)
    currentTrackedObjects = track_objects(contours)
    if len(trackedObjects) == 0:
        fillEmptyTrackedObjects(
            trackedObjects, originalFrame, currentTrackedObjects)
        continue

    objectsWithoutPair = []
    for currentTrackedObject in currentTrackedObjects:
        if currentTrackedObject.connected:
            continue
        for trackedObject in trackedObjects:
            if not trackedObject.isAlive or trackedObject.currentPairFound:
                continue
            result = trackedObject.calculateDistance(
                currentTrackedObject.centerPos)
            if result != -1:
                updateTrackedObject(
                    originalFrame, currentTrackedObject, trackedObject)

    finalizeTrackedObjects(trackedObjects)
    addNewTrackedObjects(trackedObjects, originalFrame,
                         currentTrackedObjects)
    # Draw tracks on the image
    result = draw_tracks(originalFrame, trackedObjects)
    # Display the frame (optional)
    cv2.imshow('Edge Detected Video', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
