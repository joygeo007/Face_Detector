import cv2

from random import randrange

trained_face_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Capture Video from webcam
webcam = cv2.VideoCapture(0)

#Iterate forever over frames
while True:

    #read current frame
    frame_read_bool, frame = webcam.read()

    # convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #loop for multiple faces
    for i in face_coordinates:
        (x, y, w, h) = i
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    #display img with rectangle
    cv2.imshow("Face Detector", frame)
    key = cv2.waitKey(1)

    #Stop if Q or q is pressed
    if key==81 or key==113:
        break

#release video capture object
webcam.release()
    
    





#print("Code completed")