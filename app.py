
import os

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import re
# Some utilites
import numpy as np
from util import base64_to_pil
from tensorflow.keras.preprocessing import image

input_size = 128
UPLOAD_FOLDER = 'Images'

app = Flask(__name__)

root_path = 'Models'
from tensorflow.keras.models import model_from_json
file_path = os.path.join(root_path,'model_vgg(2).json')
json_file = open(file_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
weight_file = os.path.join(root_path,'best_model_cgg.h5')
model.load_weights(weight_file)
print("Loaded model from disk")

def load_test_data(img):
    img = img.resize((128, 128))
    # Preprocessing the image
    x = image.img_to_array(img)[:,:,:3]
    img = x/255
    return img


label_map = {'agriculture': 8,
 'artisinal_mine': 1,
 'bare_ground': 6,
 'blooming': 13,
 'blow_down': 16,
 'clear': 5,
 'cloudy': 10,
 'conventional_mine': 12,
 'cultivation': 15,
 'habitation': 14,
 'haze': 11,
 'partly_cloudy': 4,
 'primary': 3,
 'road': 9,
 'selective_logging': 0,
 'slash_burn': 7,
 'water': 2}

inv_label_map = {i: l for l, i in label_map.items()}


def model_predict(img, model):
    x = load_test_data(img)
    

    preds = model.predict(np.expand_dims(x, axis=0) )[0]
    return preds


# home page
@app.route("/")
def home3():
    return render_template("base.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict3():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
    
        pred = model_predict(img,model)
        a=[]
        for j in range(17):
            if pred[j]>0.3:
                a.append(j)
        result =[]
        for p1 in a:
            t = inv_label_map[p1]
            result.append(t)         

        # Serialize the result, you can add additional fields
        return jsonify(result=result)

if __name__ == "__main__":
    app.run(port=5002,debug=False)
 
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

