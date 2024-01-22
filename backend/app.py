from flask import Flask, jsonify, request, redirect
import os
from werkzeug.utils import secure_filename
from model import ResNetModel, inference
import mlflow
from PIL import Image
from flask_cors import CORS

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


path = 'dbfs:/databricks/mlflow-tracking/2713494469571469/672969b1bd224ad49cd2c0853ea51a80/artifacts/model/'
test = mlflow.pytorch.load_model(path)

model = ResNetModel(15)
model.load_state_dict(test.state_dict())
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

    
if __name__ == '__main__':
    app.run(debug=True)
