import sys
from calendar import month

import cv2
import face_recognition
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeach
from datetime import datetime, date


from _dlib_pybind11.image_dataset_metadata import images
from pip._vendor.pygments.formatters import img

engine = textSpeach.init()


def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


path = 'student_images'
studentImg = []
studentName = []
myList = os.listdir(path)
for cl in myList:
    curimg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])


def findEncoding(images):
    imgEncodings = []
    for img in images:
        img = resize(img, 0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings




def MarkAttendence(name):
    with open('attendence.csv', 'a+') as f:

        myDatalist = f.readlines()
        nameList = []
        for line in myDatalist:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%d/%m/%Y %I:%M:%S')
            # date_object = datetime.today()
            f.writelines(f'\n{name},{timestr} ')
            #statment = str('welcome to class' + name)
            #engine.say(statment)
            #engine.runAndWait()


EncodeList = findEncoding(studentImg)

# Get a reference to webcam #0 (the default one)
vid = cv2.VideoCapture(0)


while True:
    success, frame = vid.read()
    Smaller_frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    facesInFrame = face_rec.face_locations(Smaller_frames)
    encodeFacesInFrame = face_rec.face_encodings(Smaller_frames, facesInFrame)

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame):
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (155, 155, 100), 2)
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 155, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            MarkAttendence(name)

    cv2.imshow('video', frame)
    cv2.waitKey(1)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
vid.release()
cv2.destroyAllWindows()


#Project Developed & Created by TUNK ASHWIN RAJ
#VIST https://github.com/tunkashwinraj
#FOR MORE AMAZING PROJECTS
#THANK YOU