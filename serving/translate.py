from . import app
from flask import request

def preprocess(text):
    pass

def split_lines(text):
    pass

def postprocess(text):
    pass

def merge_lines(text):
    pass

@app.route('/translate', methods=["POST"])
def translate():
    raw_text = request.json["src"]
    return {"target": raw_text}