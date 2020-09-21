#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 23:38:32 2020

@author: deviantpadam
"""


import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from model import ImageClassifier
import torch
from torchvision import transforms
from PIL import Image
import shutil

UPLOAD_FOLDER = "static/imgs"
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ImageClassifier()
model.load_state_dict(torch.load('weights',map_location='cpu'))
model.eval()
translate = {0: 'dog',
 1: 'horse',
 2: 'elephant',
 3: 'butterfly',
 4: 'chicken',
 5: 'cat',
 6: 'cow',
 7: 'sheep',
 8: 'spider',
 9: 'squirrel'}

data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def refresh():
    shutil.rmtree('static/imgs/')
    os.makedirs('static/imgs/')

def get_prediction(image_bytes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = data_transforms(image_bytes).view(1,3,224,224).to(device)
    outputs = model.forward(tensor)
    prob, y_hat = torch.topk(outputs,k=5)
    return y_hat.tolist(), prob.tolist()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    refresh()
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('predictions',
                                    filename=filename))
    return render_template('main.html')
    
@app.route('/predictions')
def predictions():
    filename = request.args.get('filename')
    image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    labels, probs = get_prediction(image)
    # print(preds)
    # print(labels[0])
    labels = [translate[i] for i in labels[0]]
    return render_template('predict.html',filename=filename, labels=labels, probs=probs[0])
    
if __name__=="__main__":
    app.run()