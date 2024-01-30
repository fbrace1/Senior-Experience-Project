import cv2
import threading
import queue
import numpy as np
import os


# Global variables
frame_queue = queue.Queue(maxsize=10)
templates = []  # Placeholder for card templates
card_counter = 0  # Counter for the card images
counter_lock = threading.Lock()  # Lock for thread-safe increments of the counter
# Global variable for templates
templates = []

def load_templates(template_folder):
    for filename in os.listdir(template_folder):
        if filename.endswith(".jpg"):
            path = os.path.join(template_folder, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # Resize the template to a fixed size
            resized_image = cv2.resize(image, (image.shape[1], image.shape[0]))
            templates.append((filename, resized_image))


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





def capture_frames(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            pass

# def compare_cards(card_image, templates, image_filename):
    # cv2.imshow(image_filename, card_image)
    #  # Save the card image to a file
    # cv2.imwrite(image_filename, card_image)
    # print(f"Card saved as {image_filename}")
    # # Placeholder for actual comparison logic

    # Define the region of interest (ROI) dimensions
def compare_cards(card_image, templates):
    # Assuming the top left corner contains the rank and suit
    roi_height, roi_width = card_image.shape[:2]
    roi_height, roi_width = int(roi_height * 0.3), int(roi_width * 0.18)

    # Crop the top-left corner and scale
    roi = card_image[0:roi_height, 0:roi_width]
    scaled_roi = cv2.resize(roi, (card_image.shape[1], card_image.shape[0]))

    card_gray = cv2.cvtColor(scaled_roi, cv2.COLOR_BGR2GRAY)
    h, w = card_gray.shape[:2]
    best_match_score = .5
    best_match_name = None

    for template_name, template in templates:
        if h <= template.shape[0] and w <= template.shape[1]:
            res = cv2.matchTemplate(card_gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_match_score:
                best_match_score = max_val
                best_match_name = template_name
            else:
                best_match_name = 'Unknown'

    return best_match_name



def main():
    load_templates('my_templates')
    cap = cv2.VideoCapture(0)
    threading.Thread(target=capture_frames, args=(cap,), daemon=True).start()

    cv2.namedWindow('Card Detector')

    while True:
        if frame_queue.empty():
            continue
        cards_detected = 0

        while not frame_queue.empty():
            frame = frame_queue.get()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        laplacian = cv2.Laplacian(adaptive_thresh, cv2.CV_64F)

        cv2.imshow('Adaptive Threshold', laplacian)

        # adaptive_thresh_gaussian = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                #  cv2.THRESH_BINARY_INV, 15, 5)
        # adaptive_thresh_gaussian = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                #  cv2.THRESH_BINARY, 11, 2)
        # adaptive_thresh_mean = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            #  cv2.THRESH_BINARY, 11, 2)
        # adaptive_thresh_mean = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            #  cv2.THRESH_BINARY_INV, 11, 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # For noise removal
        # opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        # # For closing small holes
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

        median_filtered = cv2.medianBlur(adaptive_thresh, 3)
        closing = cv2.morphologyEx(median_filtered, cv2.MORPH_CLOSE, kernel, iterations=1)

        # bilateral_filtered = cv2.bilateralFilter(adaptive_thresh, 9, 50, 50)
        # cv2.imshow('bilateral_filtered', median_filtered)

        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # show the contours (outline) of the shapes in the image    
        # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # filtered_contours = [c for c in contours if cv2.contourArea(c) > 5000 ]

        detected_cards = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if area > 4000 and 0.7 < aspect_ratio < 1.3:
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.polylines(frame, [approx.reshape(4, 2)], True, (0, 255, 0), 3)

                    
                    # # Put text above the detected card
                    # text_position = (x, y - 10) if y - 10 > 0 else (x, y + 20)
                    # cv2.putText(frame, best_match_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Compare the card and get the best match name
                   
                    # Extract the card image from the frame
                    # card_image = frame[y:y+h, x:x+w].copy()
                    # perform the perspective transform
                    warped_image = four_point_transform(frame, approx.reshape(4, 2))
                    detected_cards.append(warped_image)
                    best_match_name = compare_cards(warped_image, templates)
                    # Put text above the detected card
                    text_position = (x, y - 10) if y - 10 > 0 else (x, y + 20)
                    cv2.putText(frame, best_match_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Start a new thread for comparing this detected card to the templates
                    # threading.Thread(target=compare_cards, args=(card_image, templates), daemon=True).start()
        cv2.putText(frame, f"Detected Cards: {len(detected_cards)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Start a thread for each detected card
        for i, card_image in enumerate(detected_cards, start=1):
            image_filename = f'tempcard{i}.jpg'
            threading.Thread(target=compare_cards, args=(card_image, templates), daemon=True).start()


        cv2.imshow('Card Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
