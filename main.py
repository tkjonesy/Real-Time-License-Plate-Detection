from ultralytics import YOLO
import cv2
import re
from collections import defaultdict
import time
from paddleocr import PaddleOCR


'''
Takes an image of a cropped plate and returned the text and confidence score
returns tuple: (text, confidence)
'''
def readPlateText(plateImage, ocr):
    result = ocr.ocr(plateImage, cls=False)

    if not result[0]:
        return 'No Detection', 0

    maxArea = 0
    largestText = 'No Detection'
    largestTextConfidence = 0

    # Loop through detections
    for idx in range(len(result)):
        res = result[idx]
        for bbox, detect in res:

            # Get box Coords
            x1, y1 = bbox[0]
            x2, y2 = bbox[-2]

            # Calculate area
            area = (x2 - x1) * (y2 - y1)

            # Update area if needed
            if area > maxArea:
                maxArea = area
                largestText = detect[0].upper()
                largestTextConfidence = detect[1]

        # Remove non-alphanumeric characters
        cleanText = re.sub(r'[^a-zA-Z0-9]', '', largestText)
    return largestText, largestTextConfidence


if __name__ == '__main__':
    # Setup OCR reader and YOLO model
    ocr = PaddleOCR(lang='en')
    plateModel = YOLO('plateModel.pt')

    # Setup video capture or location of video file
    cap = cv2.VideoCapture('test-media/videotest6.mp4')

    # Calculate frame delay
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameTime = 1 / fps if fps > 0 else 1 / 30

    # Set window size
    w, h = 1280, 720
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', w, h)

    # Dictionary to track plates and their highest-confidence text
    plateTracker = defaultdict(lambda: {'text': None, 'confidence': 0, 'counter': 0})

    # Read frames
    ret = True
    while ret:
        startTime = time.time()
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

            # Read plate number with OCR
            plateText, confidence = readPlateText(platCrop, ocr)

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
                label = f'ID: {plateID} | {plateData['text']} ({plateData['confidence']:.2f})'
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # Display the processed frame
        cv2.imshow('Frame', frame)

        elapsed = time.time() - startTime
        delay = max(int((frameTime - elapsed) * 1000), 1)
        if cv2.waitKey(delay) & 0xFF == ord('q'):  # Allow quitting with 'q'
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
