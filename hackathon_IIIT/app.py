from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import sklearn
import re
import random
from random import *

app = Flask(__name__)
model = load_model('BreastNet.h5')
model2 = load_model('SkinNet.h5')
model3 = load_model('RetinaNet.h5')
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/breast')
def breast():
    return render_template('breast.html')

def breast_image(image):
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    return img

@app.route('/breast_result', methods=['POST'])
def breastMRI_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    img = breast_image(Image.open(image))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    if (prediction > 0.5).any():
        return render_template('breast.html',pred = "Breast Cancer is DETECTED. Please consult a Doctor!!!")
    else:
        return render_template('breast.html',pred = "Breast Cancer is NOT DETECTED. You are safe...")
@app.route('/skin')
def diabetes():
    return render_template('skin.html')

def skin_image(image):
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    return img

@app.route('/skin_result', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    img = skin_image(Image.open(image))
    img = np.expand_dims(img, axis=0)
    prediction = model2.predict(img)

    if (prediction > 0.5).any():
        return render_template('skin.html',pred = "Skin Cancer is DETECTED. Please consult a Doctor!!!")
    else:
        return render_template('skin.html',pred = "Skin Cancer is NOT DETECTED. You are safe...")

@app.route('/dr')
def dr():
    return render_template('dr.html')

def dr_image(image):
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    return img


@app.route('/dr_result', methods=['POST'])
def dr_result():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    img = dr_image(Image.open(image))
    img = np.expand_dims(img, axis=0)
    prediction = model3.predict(img)

    if (prediction > 0.5).any():
        return render_template('dr.html',pred = "Diabetic Retinopathy is DETECTED. Please consult a Docror!!!")
    else:
        return render_template('dr.html',pred = "Diabetic Retinopathy is NOT DETECTED. You are safe...")


if __name__ == "__main__":
    app.run(debug=True)