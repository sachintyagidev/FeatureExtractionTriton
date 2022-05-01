import werkzeug
from werkzeug.utils import cached_property, secure_filename
import flask

from flask import Flask, flash, request, redirect, url_for, send_from_directory
import json
import os
import pickle
import numpy as np
import base64

from PIL import Image as PImage
from mtcnn.mtcnn import MTCNN
import cv2 as cv

from google.cloud import vision
from google.cloud import storage
from PIL import Image, ImageDraw
from client import featurePredictions

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def featureExtraction(input_filename, model):
    feature = ''
    with open(input_filename, 'rb') as image:
        feature = featurePredictions(image, model)
    return feature

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fileFullPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fileFullPath)

            return featureExtraction(fileFullPath, request.models)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
        <label for="models">Choose a Model:</label>
        <select id="models" name="models">
            <option value="resnet50_torch">Resnet50 Torch</option>
            <option value="resnet50_onnx">Resnet50 Onnx</option>
            <option value="resnet50_trt_fp32">Resnet50 TRT fp32</option>
            <option value="resnet50_trt_fp16">Resnet50 TRT fp16</option>
            <option value="resnet50_trt_int8">Resnet50 TRT int8</option>
        </select>
        <input type=file name=file>
        <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)