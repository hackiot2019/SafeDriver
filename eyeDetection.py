import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow('img', img)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cv.waitKey(0)
cv.destroyAllWindows()
