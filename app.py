from flask import Flask, jsonify, request, redirect
import os
from werkzeug.utils import secure_filename
from model import ResNetModel, inference
import mlflow
from PIL import Image

app = Flask(__name__)
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

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({"message": "File uploaded successfully", "filename": filename})

    return jsonify({"error": "Invalid file type"})

path = 'dbfs:/databricks/mlflow-tracking/2713494469571469/672969b1bd224ad49cd2c0853ea51a80/artifacts/model/'
test = mlflow.pytorch.load_model(path)

model = ResNetModel(15)
model.load_state_dict(test.state_dict())
app.config['MODEL'] = model
@app.route('/api/predict', methods=['POST'])
def my_api_function():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    else:
        return 'Invalid file extension'
    lst_images = []
    for file in os.listdir(app.config['UPLOAD_FOLDER']):

        lst_images.append(Image.open(file))
    labels = inference(lst_images, app.config['MODEL'])
    return labels
    
if __name__ == '__main__':
    app.run(debug=True)
