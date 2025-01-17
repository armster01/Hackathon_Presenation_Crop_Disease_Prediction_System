from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
from model import CropDiseaseModel

app = Flask(__name__)

# Initialize the model
model = CropDiseaseModel()

# Load pre-trained weights if they exist
if os.path.exists('model/weights.h5'):
    model.load_weights('model/weights.h5')

# Disease classes
CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn___Cercospora_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess image
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = model.predict(image)
        predicted_class = CLASSES[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'recommendations': get_recommendations(predicted_class)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def get_recommendations(disease):
    recommendations = {
        'Apple___Apple_scab': [
            'Apply fungicide early in the season',
            'Remove infected leaves',
            'Improve air circulation by pruning'
        ],
        'Apple___Black_rot': [
            'Remove infected fruit and cankers',
            'Prune during dry weather',
            'Apply fungicides during growing season'
        ],
        # Add more recommendations for other diseases
    }
    return recommendations.get(disease, ['Consult a local agricultural expert'])

if __name__ == '__main__':
    app.run(debug=True)