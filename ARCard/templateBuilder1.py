import cv2
import os
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # top-left point will have the smallest sum, 
    # bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top-right point will have the smallest difference,
    # bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # construct the set of destination points for the "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def card_template_builder():
    # Initialize the laptop's camera
    cap = cv2.VideoCapture(0)

    # List of suits and ranks in the order you specified
    suits = ['diamonds', 'hearts', 'clubs', 'spades']
    ranks = ['ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king']

    # Create a list to keep track of the detected cards
    detected_cards = []

    # Create the output directory if it doesn't exist
    output_folder = 'my_templates'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

           # Use adaptive thresholding to handle different lighting conditions across the card
            adaptive_thresh = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY_INV, 11, 2)

            # Find contours on the adaptive threshold image
            contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
                area = cv2.contourArea(contour)

                if len(approx) == 4 and area > 5000:  # Adjust these thresholds as needed
                    cv2.polylines(frame, [approx.reshape(4, 2)], True, (0, 255, 0), 3)

                    # Perform the perspective transform
                    warped_image = four_point_transform(frame, approx.reshape(4, 2))

                    # Crop and scale the top-left corner
                    roi_height, roi_width = warped_image.shape[:2]
                    roi_height, roi_width = int(roi_height * 0.3), int(roi_width * 0.18)
                    roi = warped_image[0:roi_height, 0:roi_width]
                    scaled_roi = cv2.resize(roi, (warped_image.shape[1], warped_image.shape[0]))

                    # Generate filename and save the scaled ROI
                    suit = suits[len(detected_cards) // 13]
                    rank = ranks[len(detected_cards) % 13]
                    filename = f'{rank}_of_{suit}.jpg'
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, scaled_roi)

                    detected_cards.append((rank, suit))
                    print(f"Detected and saved {filename}")

                    prompt = input("Press ENTER to continue or 'q' to quit: ")
                    if prompt == 'q':
                        return

                    if len(detected_cards) == 52:
                        print("All cards have been detected and saved.")
                        return

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Call the function to start building card templates
card_template_builder() 