import cv2
import os

def card_template_builder():
    # Initialize the laptop's camera
    cap = cv2.VideoCapture(0)

    # Create the output directory if it doesn't exist
    output_folder = 'my_templates'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        while True:
            # Prompt the user to enter the name of the card
            card_name = input("Enter the card name (e.g., 'ace_of_spades') or 'q' to quit: ")
            if card_name == 'q':
                break
            
            card_detected = False
            while not card_detected:
                # Capture frame-by-frame
                ret, frame = cap.read()

                if not ret:
                    print("Failed to grab frame")
                    break  # If we can't get a frame, try again

                # Display the resulting frame
                cv2.imshow('frame', frame)

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

                                # Save the card image with the user-entered name
                                filepath = os.path.join(output_folder, f"{card_name}.jpg")
                                cv2.imwrite(filepath, card_image)
                                print(f"Saved {filepath}")
                                card_detected = True
                                break  # Card is saved, exit contour loop

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Allow the loop to be exited with 'q'

            if not card_detected:
                print("No card detected, please try again.")

            # Prompt to continue or quit
            if input("Do you have another card to capture? (y/n): ").lower() != 'y':
                break

    finally:
        # When everything is done, release the capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()

# Call the function to start building card templates
card_template_builder()
