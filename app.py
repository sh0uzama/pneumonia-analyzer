import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the model
MODEL_PATH = 'polmonite2.h5'  # Update this path to where you'll store the model locally

# Load model when the app starts
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
    model_loaded = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("The app will start, but predictions won't work until you provide a valid model.")
    model_loaded = False

def predict_pneumonia(file_content):
    """Process the image and return prediction results"""
    try:
        img = image.load_img(io.BytesIO(file_content), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = model.predict(img_array)[0][0]
        
        prediction_label = 'Bacterial Pneumonia' if prediction > 0.5 else 'Viral Pneumonia'
        prediction_probability = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        
        return {
            'success': True,
            'label': prediction_label,
            'probability': f"{prediction_probability:.2f}%",
            'raw_value': float(prediction)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests"""
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check server logs.'
        })
    
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file uploaded'
        })
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        })
    
    try:
        file_content = file.read()
        result = predict_pneumonia(file_content)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}'
        })

if __name__ == '__main__':
    print("\n✅ Server is running! Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True)