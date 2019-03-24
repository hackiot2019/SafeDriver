import numpy as np
import cv2 as cv
from PIL import Image
import uuid
import datetime
import pygame


def giveAnswer(image):
    return True


def playSong(a):
    if(a):
        pygame.mixer.music.unpause()
    else:
        pygame.mixer.music.pause()


def crop(image_path, coords, saved_loc):
    image_obj = Image.fromarray(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image = cropped_image.resize((200, 200))
    cropped_image.save(saved_loc)
# cropped_image.show()


face_cascade = cv.CascadeClassifier('face.xml')
eye_cascade = cv.CascadeClassifier('glass.xml')
cap = cv.VideoCapture(0)
pygame.mixer.init()
pygame.mixer.music.load("wakeMeUpChorus.mp3")
pygame.mixer.music.play()
pygame.mixer.music.pause()

while True:
    second = datetime.datetime.now().second
    i = 0
    closed = 0
    while(datetime.datetime.now().second <= second + 1):
        if(second == 59):
            second = -1
        ret, img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if (len(faces) is not 0):
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    crop(roi_gray, (ex, ey, ex+ew, ey+eh),
                         'data_data/pic' + str(i) + '.jpg')
                if(giveAnswer('data_data/pic' + str(i) + '.jpg')):
                    closed += 1
        i += 2
    if(closed/i > 0.9):
        playSong(True)
    else:
        playSong(False)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cv.waitKey(0)
cv.destroyAllWindows()
