# -*- coding: utf-8 -*-

"""
Created on Sun Sep 20 23:38:32 2020

@author: deviantpadam
"""

from flask import Flask
import os

UPLOAD_FOLDER = "app/static/imgs"


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(16)

from app import main
