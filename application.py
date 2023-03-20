import os
from datetime import datetime
from io import BytesIO

import tempfile
import atexit
import pathlib

from PIL import Image
from flask import Flask, render_template, request, send_from_directory
from transformers import ViTImageProcessor, ViTForImageClassification
from werkzeug.utils import secure_filename

application = Flask(__name__)

application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model and processor
model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
processor = ViTImageProcessor.from_pretrained('nateraw/vit-age-classifier')

age_groups = {
    0: '0-2',
    1: '3-9',
    2: '10-19',
    3: '20-29',
    4: '30-39',
    5: '40-49',
    6: '50-59',
    7: '60-69',
    8: '70-79',
    9: '80+'
}

class Prediction:
    def __init__(self, age, confidence, image_url):
        self.age = age
        self.confidence = confidence
        self.image_url = image_url
        self.timestamp = datetime.now()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def delete_temp_files(temp_dir):
    for file in temp_dir.iterdir():
        file.unlink()
    temp_dir.rmdir()

temp_dir = pathlib.Path(tempfile.mkdtemp())
atexit.register(delete_temp_files, temp_dir)

def predict_age(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    proba = outputs.logits.softmax(dim=1)
    preds = proba.argmax(dim=1)
    predicted_age = age_groups[preds.item()]
    confidence = round(proba[0][preds].item() * 100, 2)
    return predicted_age, confidence

@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')
        file = request.files['file']
        # Check if file is valid
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        if not allowed_file(file.filename):
            return render_template('index.html', error='Invalid file type')
        # Save file in temporary folder
        filename = secure_filename(file.filename)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        # Read file contents and predict age
        with open(temp_path, "rb") as image_file:
            file_bytes = image_file.read()
            age, confidence = predict_age(file_bytes)
        # Create prediction object
        image_url = f'/temp/{filename}'
        prediction = Prediction(age, confidence, image_url)
        # Render result template
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

@application.route('/temp/<filename>')
def temp_file(filename):
    return send_from_directory(temp_dir, filename)

if __name__ == '__main__':
    application.run(debug=True)