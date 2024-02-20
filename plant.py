import os
import numpy as np
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img, img_to_array

myapp = Flask(__name__)

model = load_model('model.h5')

print("loaded... Check http://127.0.0.1:5000/ ")

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(image_path):
    img = load_img(image_path, target_size=(225,225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

@myapp.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@myapp.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predictions=getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    return None


if __name__ == '__main__':
    myapp.run(debug=True)