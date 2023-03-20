import os
from datetime import datetime
from io import BytesIO

from PIL import Image
from flask import Flask, render_template, request
from transformers import ViTImageProcessor, ViTForImageClassification

application = Flask(__name__)

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
    def __init__(self, age, confidence, image_bytes):
        self.age = age
        self.confidence = confidence
        self.image_bytes = image_bytes
        self.timestamp = datetime.now()

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
        # Read file contents and predict age
        file_bytes = file.read()
        age, confidence = predict_age(file_bytes)
        # Create prediction object
        prediction = Prediction(age, confidence, file_bytes)
        # Render result template
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def predict_age(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    proba = outputs.logits.softmax(dim=1)
    preds = proba.argmax(dim=1)
    predicted_age = age_groups[preds.item()]
    confidence = round(proba[0][preds].item() * 100, 2)
    return predicted_age, confidence

if __name__ == '__main__':
    application.run(debug=True)