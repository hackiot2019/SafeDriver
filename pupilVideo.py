import cv2
import numpy as np

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.waitKey(1)
    rval, frame = vc.read()
    img = cv2.medianBlur(frame, 5)
    imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cimg = cv2.cvtColor(imgg, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(imgg, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=100)

    if circles is None:
        cv2.imshow("preview", frame)
        continue
    #circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 1)
        # draw the center of the circle
        cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("preview", frame)

    key = cv2.waitKey(20)

    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
