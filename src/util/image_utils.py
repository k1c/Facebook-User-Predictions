import cv2
from PIL import Image
import numpy as np



def detecting_faces(image):

    path_weights = "../src/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(path_weights)

    coordinates = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return coordinates


def expand(coordinates):
    pts = coordinates[0]

    x_min = pts[0]
    y_min = pts[1]
    x_max = pts[0] + pts[2]
    y_max = pts[1] + pts[3]

    x_min = max(0, x_min-(0.15 * pts[2]))
    x_max += (0.15 * pts[2])
    y_min = max(0, y_min-(0.20 * pts[3]))
    y_max += (0.20 * pts[3])
    return [[int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]]


def crop_image(image, coordinates):

    x_min = coordinates[0][0]
    x_max = coordinates[0][0] + coordinates[0][2]
    y_min = coordinates[0][1]
    y_max = coordinates[0][1] + coordinates[0][3]

    image = image[y_min:y_max, x_min:x_max]
    return image


def resize_image(image):
    image = Image.fromarray(image, 'RGB').resize((128, 128))
    return np.array(image)


def normalize_image(image):
    # image = (image - np.mean(image.reshape(-1, 3), axis=0))/255
    return image/255
