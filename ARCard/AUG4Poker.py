# Import necessary libraries
import cv2
import threading
import queue
import numpy as np
import os

# Global variables
frame_queue = queue.Queue(maxsize=10)  # Queue to hold frames from the video stream
templates = []  # Placeholder for card templates
card_counter = 0  # Counter for the card images (unused in this snippet)
counter_lock = threading.Lock()  # Lock for thread-safe increments of the counter (unused in this snippet)
# Global variables
selected_cards = []
card_positions = []

def draw_card_selection_overlay(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 255, 255)  # Yellow color in BGR format
    
    # Title text
    title = "Please select your hand"
    title_size = cv2.getTextSize(title, font, 0.5, 1)[0]
    title_x = (frame.shape[1] - title_size[0]) // 2
    title_y = frame.shape[0] // 2 - 20  # Start halfway down the screen and above the cards
    cv2.putText(frame, title, (title_x, title_y), font, 0.5, text_color, 1)
    
    # Calculate positions for each card text and draw them
    suits = ['S', 'H', 'D', 'C']  # Spades, Hearts, Diamonds, Clubs
    values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    y_start = frame.shape[0] // 2  # Starting halfway down the screen
    row_height = 30  # Height of each row
    card_positions.clear()  # Clear previous card positions if any
    
    for i, suit in enumerate(suits):
        x_start = 10  # Start from the left side for each suit
        y_position = y_start + i * row_height  # Position each suit in its own row
        for value in values:
            card_text = f"{value}{suit}"
            cv2.putText(frame, card_text, (x_start, y_position), font, 0.5, text_color, 1)
            # Update card_positions with the bounding box of the text for click detection
            text_size = cv2.getTextSize(card_text, font, 0.5, 1)[0]
            card_positions.append((x_start, y_position - text_size[1], x_start + text_size[0], y_position))
            x_start += text_size[0] + 10  # Adjust spacing based on the text size

def select_card(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, (x1, y1, x2, y2) in enumerate(card_positions):
            if x1 < x < x2 and y1 < y < y2:
                card_code = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'][idx % 13] + ['S', 'H', 'D', 'C'][idx // 13]
                if card_code not in selected_cards:
                    selected_cards.append(card_code)
                if len(selected_cards) > 2:  # Keep only the last two selections
                    selected_cards.pop(0)
                print(f"Selected Cards: {', '.join(selected_cards)}")  # Display as a comma-separated string
                break

def draw_selected_cards_on_frame(frame, selected_cards):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 255, 0)  # Green color in BGR format
    x_start = 10
    y_start = frame.shape[0] - 30  # Start drawing near the bottom of the frame
    for card in selected_cards:
        # Format the card text
        card_text = card
        cv2.putText(frame, card_text, (x_start, y_start), font, 0.7, text_color, 2)
        text_size = cv2.getTextSize(card_text, font, 0.7, 2)[0]
        x_start += text_size[0] + 10  # Move right for the next card


def show_opening_screen():
    # Create a black image
    opening_screen = np.zeros((480, 640, 3), dtype=np.uint8)
    # Set the title text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "ARcard Opening Screen"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (opening_screen.shape[1] - text_size[0]) // 2
    text_y = (opening_screen.shape[0] + text_size[1]) // 2
    # Put the title text on the black image
    cv2.putText(opening_screen, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
    # Display the opening screen
    cv2.imshow("Opening Screen", opening_screen)
    cv2.waitKey(2000)  # Display the screen for 2000 milliseconds (2 seconds)
    cv2.destroyAllWindows()

def show_menu_screen():
    # Create a black image for the menu screen
    menu_screen = np.zeros((480, 640, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(menu_screen, "Game Selection Menu", (50, 100), font, 1, (255, 255, 255), 2)

    # Game Options
    cv2.putText(menu_screen, "1. Crazy 4's", (200, 200), font, 1, (255, 255, 255), 2)
    cv2.putText(menu_screen, "2. Texas Hold'em Poker", (200, 250), font, 1, (255, 255, 255), 2)

    # Display the menu screen
    cv2.imshow("Menu", menu_screen)
    key = cv2.waitKey(0)  # Wait indefinitely for a key press

    # Check which game is selected
    if key == ord('1'):
        print("Crazy 4's Selected")
        # Here you can call a function to start Crazy 4's game
    elif key == ord('2'):
        print("Go Fish Selected")
        # Here you can call a function to start Go Fish game
    else:
        print("Invalid Selection")

    cv2.destroyAllWindows()





# Function to load card templates from a specified folder
def load_templates(template_folder):
    # Iterate through each file in the template folder
    for filename in os.listdir(template_folder):
        # Check if the file is a JPEG image
        if filename.endswith(".jpg"):
            # Construct the full path to the image
            path = os.path.join(template_folder, filename)
            # Load the image in grayscale mode
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # Append the filename and image to the templates list
            templates.append((filename, image))

# Function to order points in a clockwise manner to rectify perspective
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # Ordering the points based on their sums and differences
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# Function to apply a four point perspective transform to a region of interest
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# Function to capture frames from the video stream and put them into a queue
def capture_frames(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only add the frame to the queue if it's not full
        if not frame_queue.full():
            frame_queue.put(frame)

# Function to compare a card image against the loaded templates
def compare_cards(card_image, templates):
    # Assuming the top left corner contains the rank and suit
    roi_height, roi_width = card_image.shape[:2]
    roi_height, roi_width = int(roi_height * 0.3), int(roi_width * 0.18)

    # Crop and resize the region of interest for comparison
    roi = card_image[0:roi_height, 0:roi_width]
    scaled_roi = cv2.resize(roi, (card_image.shape[1], card_image.shape[0]))

    card_gray = cv2.cvtColor(scaled_roi, cv2.COLOR_BGR2GRAY)
    best_match_score = 0.5
    best_match_name = None

    # Iterate through each template for comparison
    for template_name, template in templates:
        # Ensure the template can be compared to the ROI
        if card_gray.shape[0] <= template.shape[0] and card_gray.shape[1] <= template.shape[1]:
            res = cv2.matchTemplate(card_gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_match_score:
                best_match_score = max_val
                best_match_name = template_name
    if best_match_name is None:
        best_match_name = 'Unknown'

    return best_match_name

# Main function to execute the card detection and recognition
def main():
    show_opening_screen()
    show_menu_screen()

    # Load the card templates
    load_templates('ARCard/my_templates')
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Start a thread to capture frames from the video stream
    threading.Thread(target=capture_frames, args=(cap,), daemon=True).start()
    
    cv2.namedWindow('Card Detector')
    cv2.setMouseCallback('Card Detector', select_card)
    
    # Wait for two cards to be selected
    while len(selected_cards) < 2:
        ret, frame = cap.read()
        if not ret:
            continue
        draw_card_selection_overlay(frame)
        cv2.imshow('Card Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clear selected card positions to prevent further selection
    card_positions.clear()
    # Main loop for processing frames
    while True:
        # Skip the loop iteration if the frame queue is empty
        if frame_queue.empty():
            continue

        # Retrieve a frame from the queue
        frame = frame_queue.get()

        draw_selected_cards_on_frame(frame, selected_cards)
        # Preprocess the frame for card detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        laplacian = cv2.Laplacian(adaptive_thresh, cv2.CV_64F)

        # Show the processed frame
        cv2.imshow('Adaptive Threshold', laplacian)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        median_filtered = cv2.medianBlur(adaptive_thresh, 3)
        closing = cv2.morphologyEx(median_filtered, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours in the processed frame
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_cards = []
        # Process each contour to identify potential cards
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check if the contour is quadrilateral and meets size criteria
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if area > 4000 and 0.7 < aspect_ratio < 1.3:
                    # Apply a perspective transform to the detected card
                    warped_image = four_point_transform(frame, approx.reshape(4, 2))
                    detected_cards.append(warped_image)
                    # Compare the transformed card image to the templates
                    best_match_name = compare_cards(warped_image, templates)
                    # Display the name of the detected card on the frame
                    text_position = (x, y - 10) if y - 10 > 0 else (x, y + 20)
                    cv2.putText(frame, best_match_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the total number of detected cards
        cv2.putText(frame, f"Detected Cards: {len(detected_cards)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with detected cards
        cv2.imshow('Card Detector', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
