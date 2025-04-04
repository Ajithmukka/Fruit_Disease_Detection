import io
import os
from django.shortcuts import render
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
import cv2
import json

label_encoder = LabelEncoder()
with open("class_names.json", "r") as f:
    class_names = json.load(f)


def classify_image(request):
    if request.method == 'POST' and request.FILES['image']:
        model_path = os.path.join('fruit_disease_detection', 'models', 'model.h5')
        model = load_model(model_path)

        # Get the uploaded image from the request
        uploaded_image = request.FILES['image']

        # Read the uploaded image as bytes and convert to a NumPy array
        image_bytes = uploaded_image.read()
        np_array = np.fromstring(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (128, 128))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch size
        prediction = model.predict(image)
        predicted_label = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted class

        disease_name = class_names[predicted_label]
        prediction_label = disease_name

        return render(request, 'result.html', {'prediction': prediction_label})

    return render(request, 'upload.html')
