import io

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS

from models import Avatar_Generator_Model
from utils import *

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
CONFIG_FILENAME = "config.json"
DOWNLOAD_DIRECTORY = None

app = Flask(__name__)
CORS(app)

MODEL = None


def is_file_allowed(filename):
    """
    Validates provided file extension. Allowed extensions are jpg, jpeg and png.
    """

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def bad_request(message):
    response = jsonify({'message': message})
    response.status_code = 400

    return response


def face_to_cartoon(DOC_FILE, face):
    """
    Converts face image into cartoon.
    """

    document_name = DOC_FILE.split('.')[0]
    extension = (DOC_FILE.split('.')[-1]).lower()
    document = Image.open(io.BytesIO(face))

    if not os.path.exists(DOWNLOAD_DIRECTORY):
        os.makedirs(DOWNLOAD_DIRECTORY)

    if extension == "png":
        format_image = "PNG"
    else:
        extension = "jpg"
        format_image = "JPEG"

    filename_face = f"{document_name}.{extension}"
    document.save(DOWNLOAD_DIRECTORY + filename_face, format_image, quality=80, optimize=True, progressive=True)

    cartoon_filename = f"{document_name}_cartoon.jpg"
    MODEL.generate(DOWNLOAD_DIRECTORY + filename_face, DOWNLOAD_DIRECTORY + cartoon_filename)

    return cartoon_filename


@app.route('/generate', methods=['POST'])
def predict():
    """
    REST API endpoint that accepts 'image' file as form-data and returns generated cartoon image.
    """

    if 'image' not in request.files:
        return bad_request('File is missing.')

    file = request.files['image']
    if file is None:
        return bad_request('File is missing.')

    if is_file_allowed(file.filename):
        cartoon_filename = face_to_cartoon(file.filename, file.read())
        try:
            return send_from_directory(DOWNLOAD_DIRECTORY, filename=cartoon_filename, as_attachment=True)
        except FileNotFoundError:
            abort(404)
    else:
        return bad_request('File type unsupported.')

    response = jsonify({'message': 'Unexpected error occurred.'})
    response.status_code = 500

    return response


if __name__ == "__main__":
    config = configure_model(CONFIG_FILENAME, use_wandb=False)
    DOWNLOAD_DIRECTORY = config.download_directory

    MODEL = Avatar_Generator_Model(config, use_wandb=False)
    MODEL.load_weights(config.model_path)

    app.run(host="0.0.0.0", port="9999")
