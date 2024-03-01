# Import necessary libraries
import cv2
import threading
import queue
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\FBRAC\Projects\FredSeniorExperiment\Senior-Experience-Project\250epochs_conv.h5')
class_names = [
    'Ace of Diamonds', 'Two of Diamonds', 'Three of Diamonds', 'Four of Diamonds', 'Five of Diamonds', 
    'Six of Diamonds', 'Seven of Diamonds', 'Eight of Diamonds', 'Nine of Diamonds', 'Ten of Diamonds', 
    'Jack of Diamonds', 'Queen of Diamonds', 'King of Diamonds',
    'Ace of Hearts', 'Two of Hearts', 'Three of Hearts', 'Four of Hearts', 'Five of Hearts', 
    'Six of Hearts', 'Seven of Hearts', 'Eight of Hearts', 'Nine of Hearts', 'Ten of Hearts', 
    'Jack of Hearts', 'Queen of Hearts', 'King of Hearts',
    'Ace of Clubs', 'Two of Clubs', 'Three of Clubs', 'Four of Clubs', 'Five of Clubs', 
    'Six of Clubs', 'Seven of Clubs', 'Eight of Clubs', 'Nine of Clubs', 'Ten of Clubs', 
    'Jack of Clubs', 'Queen of Clubs', 'King of Clubs',
    'Ace of Spades', 'Two of Spades', 'Three of Spades', 'Four of Spades', 'Five of Spades', 
    'Six of Spades', 'Seven of Spades', 'Eight of Spades', 'Nine of Spades', 'Ten of Spades', 
    'Jack of Spades', 'Queen of Spades', 'King of Spades',
    'Joker', 'Joker', 'Back Card'
]


# Global variables
frame_queue = queue.Queue(maxsize=10)  # Queue to hold frames from the video stream
templates = []  # Placeholder for card templates
card_counter = 0  # Counter for the card images (unused in this snippet)
counter_lock = threading.Lock()  # Lock for thread-safe increments of the counter (unused in this snippet)


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
    cv2.putText(menu_screen, "2. Go Fish", (200, 250), font, 1, (255, 255, 255), 2)

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
def compare_cards(card_image, model):
    # Convert the card image to grayscale
    card_image_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    # Resize the image to match the input shape expected by the model
    card_image_resized = cv2.resize(card_image_gray, (180, 180))
    # Expand the dimensions of the image to match the input shape expected by the model
    card_image_expanded = np.expand_dims(card_image_resized, axis=[0, -1])
    # Normalize the pixel values
    card_image_expanded = card_image_expanded / 255.0
    # Make a prediction
    predictions = model.predict(card_image_expanded)
    # Get the index of the highest probability class
    predicted_class = np.argmax(predictions)
    # Map the predicted class index to the corresponding class name
    best_match_name = class_names[predicted_class]  # You need to define 'class_names' list with the names of your classes
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
    cv2.resizeWindow('Card Detector', 1080, 720)

    # Main loop for processing frames
    while True:
        # Skip the loop iteration if the frame queue is empty
        if frame_queue.empty():
            continue

        # Retrieve a frame from the queue
        frame = frame_queue.get()

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
                    best_match_name = compare_cards(warped_image, model)
                    # Display the name of the detected card on the frame
                    text_position = (x, y - 10) if y - 10 > 0 else (x, y + 20)
                    cv2.putText(frame, best_match_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the total number of detected cards
        cv2.putText(frame, f"Detected Cards: {len(detected_cards)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        resized_frame = cv2.resize(frame, (1080, 720))
        # Show the frame with detected cards
        cv2.imshow('Card Detector', resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
