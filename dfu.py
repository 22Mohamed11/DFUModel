import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import keras

app = Flask(__name__)

model1 = keras.models.load_model('teath.keras')
class_labels1 = ['Caries', 'Gingivitis', 'Hypodontia', 'Mouth Ulcer', 'Tooth_discoloration_augmented']

model2 = keras.models.load_model('dfu.keras')
class_labels2 = ['DFU', 'Wound']

def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/teath', methods=['POST'])
def predict_teath():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_image(image)

        predictions1 = model1.predict(img_array)
        predicted_class_index1 = np.argmax(predictions1)
        predicted_class_label1 = class_labels1[predicted_class_index1]
        class_probabilities1 = predictions1[0].tolist()

        return jsonify({
            'predicted_class': predicted_class_label1,
            'class_probabilities': dict(zip(class_labels1, class_probabilities1))
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/dfu', methods=['POST'])
def predict_dfu():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_image(image)

        predictions2 = model2.predict(img_array)
        predicted_class_index2 = np.argmax(predictions2)
        predicted_class_label2 = class_labels2[predicted_class_index2]
        class_probabilities2 = predictions2[0].tolist()

        return jsonify({
            'predicted_class': predicted_class_label2,
            'class_probabilities': dict(zip(class_labels2, class_probabilities2))
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7000, debug=True)
