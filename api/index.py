import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__, template_folder="../templates", static_folder="../static")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "DNBSD.h5")

model = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        model = get_model()

        img = Image.open(temp_path).convert("RGB")
        img = img.resize((256, 256))
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        prediction = model.predict(x)
        score = float(prediction[0][0])

        label = "Stroke" if score > 0.5 else "Normal"
        confidence = score if score > 0.5 else 1 - score

        return jsonify({
            "label": label,
            "confidence": f"{confidence * 100:.2f}%",
            "raw_score": score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
