from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from PIL import Image
import pandas as pd
from keras.models import load_model




app = Flask(__name__)
model = load_model('model.h5')

info = pd.read_excel('info.xlsx')


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_files = request.files.getlist('file[]')
    l = len(uploaded_files)
    for i, file in enumerate(uploaded_files):
        img = Image.open(file)
        if img.format != 'JPG':
            img = img.convert('RGB')
        img.save(f'static/uploaded_image_{i}.jpg')
    pred_values = []
    for i in range(l):
        img = np.asarray(Image.open(
            f'static/uploaded_image_{i}.jpg').resize((28, 28)))
        img = img/255
        i_array = np.array([img])
        key = np.argmax(model.predict(i_array))
        pred_values.append(key)

    pred_v = pd.DataFrame(pred_values, columns=['values'])
    disease = pred_v['values'].mode()
    mode_disease = int(disease.values[0])
    disease_name = list(info[info['Index'] == mode_disease]['Disease Name'])[0]
    basic_info = list(info[info['Index'] == mode_disease]
                      ['Basic Information'])[0]
    Common_name = list(info[info['Index'] == mode_disease]['Common Name'])[0]
    Symptoms = list(info[info['Index'] == mode_disease]['Symptoms'])[0]
    Treatment = list(info[info['Index'] == mode_disease]['Treatment'])[0]
    for i in range(l):
        os.remove(f'static/uploaded_image_{i}.jpg')
    return render_template('submit.html', title=disease, desc_name=disease_name, basic_info=basic_info,
                           Common_name=Common_name, Treatment=Treatment, Symptoms=Symptoms)


if __name__ == '__main__':
    app.run(debug=True)
