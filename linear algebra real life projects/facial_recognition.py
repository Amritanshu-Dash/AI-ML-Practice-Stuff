import cv2
import numpy as np
import matplotlib.pyplot as plt

# loading load faces
known_faces = {
    "galaxy": cv2.imread("linear algebra real life projects/dark galaxy background, glowing cyan vector arrows, neural network layer design connections, minimalist tech realted to maths and artificial intelligence, 1584×396 pixels.jpg", 0),
    "car": cv2.imread("linear algebra real life projects/image-05.jpg", 0)
}

#train the model or recognizer (LBPH = Local Binary Patterns Histograms)
recognizer = cv2.face.LBPHFaceRecognizer_create()
images = []
labels = []

for i, (name, img) in enumerate(known_faces.items()):
    images.append(img)
    labels.append(i)

recognizer.train(images, np.array(labels))


