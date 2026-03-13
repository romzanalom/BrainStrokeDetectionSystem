import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'DNBSD.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable
model = None

def load_brain_stroke_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading model from {MODEL_PATH}...")
            model = load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a dummy model for demonstration purposes (since file might be corrupt or incompatible).")
            # Fallback to creating a dummy architecture if load fails (optional, mostly for dev)
            create_dummy_model()
    else:
        print(f"Model file {MODEL_PATH} not found. functionality will be limited.")

def create_dummy_model():
    # This is just to prevent crash if model is missing during dev, 
    # but practically we need the real model.
    # We will just print a warning in the predict route if model is None.
    pass

load_brain_stroke_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if model is None:
             # Try reloading
            load_brain_stroke_model()
            if model is None:
                return jsonify({'error': 'Model not loaded. Please place DNBSD.h5 in the root directory.'})

        try:
            # Preprocessing matching the training steps
            # img_width, img_height = 256, 256
            # rescale=1.0/255.0
            
            from PIL import Image
            img = Image.open(filepath).convert('RGB')
            img = img.resize((256, 256))
            x = np.array(img, dtype=np.float32)
            x = x / 255.0  # Rescale like in training
            x = np.expand_dims(x, axis=0)

            prediction = model.predict(x)
            # prediction is sigmoid output (0 to 1)
            # user said: 0 for normal and 1 for stroke
            
            score = float(prediction[0][0])
            label = "Stroke" if score > 0.5 else "Normal"
            confidence = score if score > 0.5 else 1 - score
            
            return jsonify({
                'label': label,
                'confidence': f"{confidence*100:.2f}%",
                'raw_score': score
            })
        except Exception as e:
            return jsonify({'error': str(e)})
            
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
