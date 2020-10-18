import requests
import numpy as np
from PIL import Image
from io import BytesIO

from tensorflow.keras.models import load_model
from helpers import download_model

model_url = "https://www.dropbox.com/s/sb2z0nclqtgg9tk/model_3class.zip?raw=1"
model_path = 'best_model_3class.hdf5'
labels_path = "labels.txt"


class syndicai(object):
    def __init__(self):
        download_model(model_url)
        self.model = load_model(model_path, compile=False)

    def predict(self, url, features_names=None):
        image = requests.get(url).content
        # adjust the image size to the size of images in the train dataset
        image_preprocessed = np.array(Image.open(BytesIO(image)).resize((299, 299))).astype(np.float) / 255
        image_preprocessed = np.expand_dims(image_preprocessed, axis=0)
        # load labels
        with open(labels_path) as file:
            labels = file.read()
        labels = labels.split(',')
        predictions = self.model.predict(image_preprocessed)
        index = np.argmax(predictions)
        return labels[index]

    def metrics(self):
        return 'Not Implemented'
