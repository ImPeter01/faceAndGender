from keras_preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# load model
model = load_model('gender_detect.model')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')

# open webcam
webcam = cv2.VideoCapture(0)

classes = ['man','woman']
# loop through frames
if not webcam.isOpened():
    print("Cannot open webcam")
while webcam.isOpened():

    # read frame from webcam
    _,frame = webcam.read()
    frame_cpy = frame.copy()
    frame_cpy = cv2.cvtColor(frame_cpy, cv2.COLOR_BGR2GRAY)

    # apply face detection
    face = face_cascade.detectMultiScale(
        frame_cpy,
        scaleFactor=1.2,
        minNeighbors=2,
        minSize=(30, 30))

    # loop through detected faces
    for idx, f in enumerate(face):
        print("processing detect face")
        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        print("startX", startX)
        print("startY", startY)
        print("endX", endX)
        print("endY", endY)

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (startX+endX,startY+endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = frame[startY:(startY+endY),startX:(startX+endX)]
        print("shape[0]", face_crop.shape[0])
        print("shape[1]", face_crop.shape[1])
        print("shape[2]", face_crop.shape[2])

        if (face_crop.shape[0]) < 30 or (face_crop.shape[1]) < 30:
            continue

        # draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (startX + endX, startY + endY), (0, 255, 0), 2)

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop,batch_size= 32) # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        # get label with max accuracy
        idx = np.argmax(conf)
        print("result data:",idx)
        label = classes[idx]
        percents = conf[0][idx] * 100

       # label = conf[idx] * 100 + "," + label
        label = "{gender},{percent}".format(gender=label, percent="{:.2f}%".format(percents))
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
