import numpy as np
import cv2 as cv
from PIL import Image
import uuid


def crop(image_path, coords, saved_loc):
    image_obj = Image.fromarray(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image = cropped_image.resize((50, 50))
    cropped_image.save(saved_loc)
    # cropped_image.show()


face_cascade = cv.CascadeClassifier('face.xml')
eye_cascade = cv.CascadeClassifier('glass.xml')
cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if (len(faces) is not 0):
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                crop(roi_gray, (ex, ey, ex+ew, ey+eh),
                     'test_data/closed' + str(uuid.uuid4()) + '.jpg')
    cv.imshow('img', img)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cv.waitKey(0)
cv.destroyAllWindows()
