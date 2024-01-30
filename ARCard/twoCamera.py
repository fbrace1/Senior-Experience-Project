import cv2

# Initialize the first webcam
cap1 = cv2.VideoCapture(0)  # 0 is usually the default camera

# Initialize the second webcam. Change the index if needed.
cap2 = cv2.VideoCapture(1)  # 1 is often the second camera

while True:
    # Capture frame-by-frame from both webcams
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Check if frame is read correctly from both webcams
    if not ret1 or not ret2:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frames
    cv2.imshow('Webcam 1', frame1)
    cv2.imshow('Webcam 2', frame2)

    # Break the loop with 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release everything when done
cap1.release()
cap2.release()
cv2.destroyAllWindows()
