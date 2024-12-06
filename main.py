from ultralytics import YOLO
import cv2
import re
import easyocr
from collections import defaultdict


'''
Takes an image of a cropped plate and returned the text and confidence score
returns tuple: (text, confidence)
'''
def readPlateText(plate, reader):
    result = reader.readtext(plate)

    maxArea = 0
    largestText = None
    largestTextConfidence = 0

    for bbox, text, confidence in result:
        # Get box Coords
        x1, y1 = bbox[0]
        x2, y2 = bbox[-2]

        # Calculate area
        area = (x2 - x1) * (y2 - y1)

        # Update area if needed
        if area > maxArea:
            maxArea = area
            largestText = text.upper()
            largestTextConfidence = confidence

    if not largestText or largestTextConfidence < .2:
        return None, 0

    # Remove non-alphanumeric characters
    cleanText = re.sub(r'[^a-zA-Z0-9]', '', largestText)

    return cleanText, largestTextConfidence


if __name__ == "__main__":
    # Setup OCR reader
    reader = easyocr.Reader(['en'])

    # baseModel = YOLO('yolo11m.pt')
    plateModel = YOLO('plateModel.pt')

    # Setup video capture or location of video file
    cap = cv2.VideoCapture('test-media/videotest.mp4')

    # Dictionary to track plates and their highest-confidence text
    plateTracker = defaultdict(lambda: {'text': None, 'confidence': 0, 'counter': 0})

    # Read frames
    ret = True
    while ret:
        ret, frame = cap.read()

        if not ret:
            print('Failed to read frame')
            break

        # Check for tag
        detections = plateModel.track(frame, conf=0.7, persist=True, verbose=False)[0]

        # IDs present
        currIDs = set()

        # Loop through plates
        for detection in detections.boxes.data.tolist():
            if len(detection) != 7:
                continue
            x1, y1, x2, y2, id, score, _ = detection
            currIDs.add(id)

            # Crop image
            platCrop = frame[int(y1):int(y2), int(x1):int(x2)].copy()

            # Filter cropped plate
            platCropGrey = cv2.cvtColor(platCrop, cv2.COLOR_BGR2GRAY)

            # Resize the image to 8x
            scale = 8
            resizedImage = cv2.resize(platCropGrey, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Otsu thresholding
            _, plateThresh = cv2.threshold(platCropGrey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            plateThresh = cv2.morphologyEx(plateThresh, cv2.MORPH_CLOSE, kernel)

            # Read plate number with OCR
            plateText, confidence = readPlateText(plateThresh, reader)

            # Update tracked plate information if confidence is higher
            if plateText and confidence > plateTracker[id]['confidence']:
                plateTracker[id]['text'] = plateText
                plateTracker[id]['confidence'] = confidence
                plateTracker[id]['counter'] = 0

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Update counters for timeout
        idToRemove = []
        for id in plateTracker:
            if id not in currIDs:
                plateTracker[id]['counter'] += 1
                if plateTracker[id]['counter'] > 5:
                    idToRemove.append(id)

        # Remove expired IDs
        for id in idToRemove:
            del plateTracker[id]

        # Display the highest-confidence text for each tracked plate
        for plateID, plateData in plateTracker.items():
            if plateData['text']:
                label = f"ID: {plateID} Text: {plateData['text']} ({plateData['confidence']:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

        # Display the processed frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):  # Allow quitting with 'q'
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
