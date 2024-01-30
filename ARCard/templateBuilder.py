import cv2
import os

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
                # Approximate the contour to a polygon
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

                # Check for rectangular shape (which a card would likely have)
                if len(approx) == 4:
                    # Check if the area of the rectangle is large enough to be a card
                    area = cv2.contourArea(contour)
                    if area > 5000:  # Adjust this threshold based on the actual size
                        # Extract and save the card
                        x, y, w, h = cv2.boundingRect(contour)

                        # Make sure we extract only within the frame bounds
                        if x >= 0 and y >= 0 and (x+w) <= frame.shape[1] and (y+h) <= frame.shape[0]:
                            card_image = frame[y:y+h, x:x+w]

                            # Generate the filename based on the current count of detected cards
                            suit = suits[len(detected_cards) // 13]
                            rank = ranks[len(detected_cards) % 13]
                            filename = f'{rank}_of_{suit}.jpg'
                            filepath = os.path.join(output_folder, filename)
                            cv2.imwrite(filepath, card_image)

                            # Add the card to the list of detected cards
                            detected_cards.append((rank, suit))

                            print(f"Detected and saved {filename}")
                            prompt = input("Press ENTER to continue or 'q' to quit: ")
                            if prompt == 'q':
                                return

                            # Check if we have detected all cards
                            if len(detected_cards) == 52:
                                print("All cards have been detected and saved.")
                                return  # Exit the function after saving all cards

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Break the loop with the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

# Call the function to start building card templates
card_template_builder()
