import cv2

from random import randrange

trained_face_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# img to detect faces in
img = cv2.imread("stickman.png")

# convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#loop for multiple faces
for i in face_coordinates:
    (x, y, w, h) = i
    # Draw rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 4)

#display img with rectangle
cv2.imshow("Face Detector", img)
cv2.waitKey()


print("Code completed")