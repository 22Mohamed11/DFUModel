from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import keras

app = Flask(__name__)

# Load the pre-trained model DFU
model = keras.models.load_model('dfu.keras')
class_labels = ['DFU', 'Wound']


def preprocess_image(image):
    img = image.resize((150, 150))  # Resize the image to match model input size
    img_array = np.array(img)  # Convert PIL image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis = 0)  # Add batch dimension
    return img_array

#predict DFU
@app.route('/dfu', methods=['POST'])

def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            img_array = preprocess_image(image)

            # Make predictions using the loaded model
            predictions = model.predict(img_array)

            # Get the predicted class index
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]

            # Get class probabilities
            class_probabilities = predictions[0].tolist()

            return jsonify({
                'predicted_class': predicted_class_label,
                'class_probabilities': dict(zip(class_labels, class_probabilities))
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 7000 ,debug = True)