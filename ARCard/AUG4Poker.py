# Import necessary libraries
import cv2
import threading
import queue
import numpy as np
import os

# from screeninfo import get_monitors

# Global variables
frame_queue = queue.Queue(maxsize=10)  # Queue to hold frames from the video stream
templates = []  # Placeholder for card templates
card_counter = 0  # Counter for the card images (unused in this snippet)
counter_lock = threading.Lock()  # Lock for thread-safe increments of the counter (unused in this snippet)
# Global variables
selected_cards = []
card_positions = []




def on_crazy_4s_click(*args):
    print("Crazy 4's Selected")
    # Call the function to start Crazy 4's game here

def on_texas_holdem_click(*args):
    print("Texas Hold'em Selected")
    detect_cards_poker()

def on_go_fish_click(*args):
    print("Go Fish Selected")
    # Call the function to start Go Fish game here

def on_blackjack_click(*args):
    print("Blackjack Selected")
    # Call the function to start Blackjack game here

def on_quit_click(*args):
    print("Quit Selected")
    # Call the function to quit the game here


def draw_card_selection_overlay(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 255, 200)  # Yellow color in BGR format
    
    # Title text
    title = "TEXAS HOLD'EM POKER"
    heading = "Please select the cards that you have been dealt: (your hand)"
    title_size = cv2.getTextSize(title, font, 0.5, 1)[0]
    title_x = 10#(frame.shape[1] - title_size[0]) // 2
    title_y = 70#frame.shape[0] // 2 - 20  # Start halfway down the screen and above the cards
    cv2.putText(frame, title, (title_x, title_y), font, 0.8, text_color, 1)
    cv2.putText(frame, heading, (title_x, title_y+20), font, 0.6, (0,255,150), 1)
    
    # Calculate positions for each card text and draw them
    suits = ['S', 'H', 'D', 'C']  # Spades, Hearts, Diamonds, Clubs
    values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    row_height = 30  # Height of each row
    card_positions.clear()  # Clear previous card positions if any

    # Calculate the starting y position based on the number of rows and row height
    num_rows = len(suits)
    y_start = frame.shape[0] - (num_rows * row_height) - 20  # 20 is a bottom margin

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
    # Get the size of the primary monitor
    # monitor = get_monitors()[0]
    # screen_width = monitor.width
    # screen_height = monitor.height

    # Create an opening screen with the size of the monitor
    # opening_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
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
    cv2.putText(menu_screen, "Game Selection Menu", (50, 50), font, 1, (255, 255, 255), 2)

    # Game Options
    options = ["1. Crazy 4's", "2. Texas Hold'em Poker", "3. Go Fish", "4. Blackjack", "5. Quit"]
    option_positions = []

    for i, option in enumerate(options):
        y_position = 200 + i * 50
        cv2.putText(menu_screen, option, (100, y_position), font, 1, (255, 255, 255), 2)
        text_size = cv2.getTextSize(option, font, 1, 2)[0]
        option_positions.append((200, y_position - text_size[1], 200 + text_size[0], y_position))

    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (x1, y1, x2, y2) in enumerate(option_positions):
                if x1 < x < x2 and y1 < y < y2:
                    print(f"{options[i]} Selected")
                    # Call the corresponding function for the selected game here
                    cv2.destroyAllWindows()
                    if i == 0:
                        on_crazy_4s_click()
                    elif i == 1:
                        on_texas_holdem_click()
                    elif i == 2:
                        on_go_fish_click()
                    elif i == 3:
                        on_blackjack_click()
                    elif i == 4:
                        on_quit_click()
                    
    cv2.imshow("Menu", menu_screen)
    cv2.setMouseCallback("Menu", on_mouse_click)
    cv2.waitKey(0)

def calculate_poker_probabilities(hand_cards, detected_cards=[]):
    # Placeholder function for calculating poker probabilities
    # For now, it returns dummy probabilities
    return {
        'Royal Flush': '0.000154%',
        'Straight Flush': '0.00139%',
        'Four of a Kind': '0.0240%',
        'Full House': '0.144%',
        'Flush': '0.197%',
        'Straight': '0.392%',
        'Three of a Kind': '2.11%',
        'Two Pair': '4.75%',
        'One Pair': '42.3%',
        'High Card': '50.1%'
    }

def generate_probabilities_image(poker_hands):
    probabilities_image = np.zeros((400, 300, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    line_type = 1
    start_y = 20
    line_height = 20

    for i, (hand, probability) in enumerate(poker_hands.items()):
        text = f"{hand}: {probability}"
        cv2.putText(probabilities_image, text, (10, start_y + i * line_height), font, font_scale, color, line_type)

    return probabilities_image



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


def detect_cards_poker():
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
        #show the probability window with probabilities
        poker_probabilities = calculate_poker_probabilities(selected_cards)
        probabilities_image = generate_probabilities_image(poker_probabilities)
        cv2.imshow('Probabilities', probabilities_image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
# Main function to execute the card detection and recognition
def main():
    show_opening_screen()
    show_menu_screen()


if __name__ == "__main__":
    main()
