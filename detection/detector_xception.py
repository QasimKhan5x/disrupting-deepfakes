import json
import os
import shutil
import warnings

import cv2
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import torch
import torch.nn as nn
from detect_from_video import *
from network.models import *
from network.xception import *
from sklearn.metrics import log_loss
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

face_detector = dlib.get_frontal_face_detector()

# Text variables
font_face = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2
font_scale = 1

xception_model_path = '/home/data/FaceForensics++/models/faceforensics++_models_subset/face_detection/xception/all_c23.p'
xception_model, *_ = model_selection(modelname='xception', num_out_classes=2)
xception_model = torch.load(xception_model_path, map_location = device)
torch.save(xception_model, "xception/weights.ckpt")


# filter out the SourceChangeWarning
warnings.filterwarnings("ignore")

def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Model prediction
    output = model(preprocessed_image.to(device))
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def extract_face(input):

    image = input
    
    height, width = image.shape[:2]
    
    # Detect with dlib
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces):
        # For now only take biggest face
        face = faces[0]

        # --- Prediction ---------------------------------------------------
        # Face crop with dlib and bounding box scale enlargement
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = image[y:y+size, x:x+size]
        return cropped_face, face

def Xception_Detector2D(org_img):
    model = xception_model
    model = model.to(device)

    try:
        input, face = extract_face(org_img)
    except:
        input = org_img
        face = None
    prediction, output = predict_with_model(input, model,
                                                cuda=True)
    return prediction, output
    
    
    

