import torch
from flask import Flask, jsonify, request, redirect
import os
from werkzeug.utils import secure_filename
from model import ResNetModel, inference
import mlflow
from PIL import Image
from flask_cors import CORS
import json
from kafka import KafkaProducer
import uuid
import pymongo


# Kafka configuration
KAFKA_BROKER_URL = '51.178.53.42:9092'  # Modify with your Kafka broker address
KAFKA_TOPIC = 'Human_action'  # Modify with your Kafka topic name

# Set up Kafka producer
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER_URL],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

client = pymongo.MongoClient("mongodb://localhost:27017/")  # Modify with your MongoDB connection string
db = client["human-action"]  # Replace with your database name

app = Flask(__name__)
CORS(app)
# Configure the maximum upload size (for example, 16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure there is a folder to save the uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Ensure there is a folder to save the uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions (for example, only images)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


path = 'model.pth'

model = ResNetModel(15)
model.load_state_dict(torch.load(path))
app.config['MODEL'] = model

@app.route('/api/predict', methods=['POST'])
def my_api_function():

    files = request.files.getlist('files')

    if not files:
        return jsonify({"error": "No files selected"})

    image_list = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_list.append(Image.open(filepath))
        else:
            return jsonify({"error": "Invalid file type"})

    if not image_list:
        return jsonify({"error": "No valid images processed"})

    labels = inference(image_list, app.config['MODEL'])

    # Optional: Clean up uploaded files after processing
    for image in image_list:
        os.remove(image.filename)

    return jsonify(labels)


@app.route('/api/send-to-kafka', methods=['POST'])
def send_to_kafka():
    file = request.files['file']
    if file and allowed_file(file.filename):
        original_filename, file_extension = os.path.splitext(secure_filename(file.filename))

        # Keep generating new filenames until a unique one is found
        while True:
            unique_id = uuid.uuid4().hex
            new_filename = f"{original_filename}_{unique_id}{file_extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

            # Check if the new filename already exists
            if not os.path.exists(filepath):
                break  # Unique filename found, exit the loop

        # Save the file with the unique filename
        file.save(filepath)

        # Send the new filename to Kafka
        message = {'filename': new_filename, 'message': 'Image received'}
        producer.send(KAFKA_TOPIC, value=message)

        return jsonify({"message": "Image sent to Kafka successfully", "id":new_filename}), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/acquire/<name_id>', methods=['GET'])
def acquire(name_id):
    res = db.results.find_one({'filename': name_id})
    del res['_id']
    return jsonify(res), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
