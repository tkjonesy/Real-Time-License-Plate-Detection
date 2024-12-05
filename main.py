from ultralytics import YOLO
import cv2


'''
Takes in a string and detects whether or not it follows a certain pattern (Florida plates)
'''
def checkPlateFormat(text):
    return False


'''
Takes an image of a cropped plate and returned the text and confidence score
returns tuple: (text, confidence)
'''
def readPlateText(plate):

    return 0, 0


if __name__ == "__main__":
    baseModel = YOLO('yolo11m.pt')
    # plateModel = YOLO('best.pt')

    # Setup video capture
    cap = cv2.VideoCapture(0)

    # Read frames
    ret = True
    i = 0
    while ret:
        if i > 10:
            break
        ret, frame = cap.read()

        if ret:
            pass

        # Check for tag
        detections = baseModel(frame)[0]
        # Loop through plates
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, id = detection

            # Crop image
            platCrop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # Filter cropped plate
            platCropGrey = cv2.cvtColor(platCrop.copy(), cv2.COLOR_BGR2GRAY)
            _, plateThresh = cv2.threshold(platCropGrey.copy(), 64, 255, cv2.THRESH_BINARY_INV)

            # Read plate number with OCR
            platetext, confidence = readPlateText(plateThresh)

            # Write text to bounding box

        i += 1
