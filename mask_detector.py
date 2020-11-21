import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils


def detecting_mask(frame, net, loaded_model):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (256, 256), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    faces = []
    locations = []
    predictions = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (256, 256))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locations.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype = 'float32')
        predictions = loaded_model.predict(faces, batch_size = 32)

    return (locations, predictions)


print('[INFO] Loading Face Detector Model...')
prototxt_path = r'face_detector\deploy.prototxt'
weights_path = r'face_detector\res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNet(prototxt_path, weights_path)

loaded_model = load_model('Facemask_Detector.model')

print('[INFO] Starting Video Stream....')
vs = VideoStream(src=0).start()


while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    (locations, predictions) = detecting_mask(frame, net, loaded_model)

    for (box, prediction) in zip(locations, predictions):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = prediction

        label = 'Mask' if mask > withoutMask else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

        label = '{}: {:.2f}%'.format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow('Facemask Detector', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


cv2.destroyAllWindows()
vs.stop()
