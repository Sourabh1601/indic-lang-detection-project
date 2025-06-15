import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Suppress GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import numpy as np
import librosa
import pickle
import tensorflow as tf  # For TFLite
import logging
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.static_folder = 'static'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3'}  # Support only .wav and .mp3
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize model variables as None
models_dir = os.path.join(BASE_DIR, 'models')
rf_model = None
svm_model = None
ann_model = None
label_encoder = None
scaler = None

# Load Random Forest model
rf_path = os.path.join(models_dir, 'rf_model.pkl')
if os.path.exists(rf_path):
    try:
        with open(rf_path, 'rb') as f:
            rf_model = pickle.load(f)
        logger.info(f"Random Forest model loaded successfully from {rf_path}")
    except Exception as e:
        logger.error(f"Error loading Random Forest model from {rf_path}: {str(e)}")
else:
    logger.warning(f"Random Forest model file not found at: {rf_path}")

# Load SVM model
svm_path = os.path.join(models_dir, 'svm_model.pkl')
if os.path.exists(svm_path):
    try:
        with open(svm_path, 'rb') as f:
            svm_model = pickle.load(f)
        logger.info("SVM model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading SVM model: {str(e)}")
else:
    logger.warning(f"SVM model file not found at: {svm_path}")

# Load ANN model as TFLite
ann_path = os.path.join(models_dir, 'ann_model.tflite')
if os.path.exists(ann_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=ann_path)
        interpreter.allocate_tensors()
        ann_model = interpreter
        logger.info("TFLite ANN model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading TFLite ANN model: {str(e)}")
        ann_model = None
else:
    logger.warning(f"TFLite ANN model file not found at: {ann_path}")
    ann_model = None

# Load label encoder
label_encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
if os.path.exists(label_encoder_path):
    try:
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info("Label encoder loaded successfully")
    except Exception as e:
        logger.error(f"Error loading label encoder: {str(e)}")
else:
    logger.warning(f"Label encoder file not found at: {label_encoder_path}")

# Load scaler
scaler_path = os.path.join(models_dir, 'scaler.pkl')
if os.path.exists(scaler_path):
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")
    except Exception as e:
        logger.error(f"Error loading scaler: {str(e)}")
else:
    logger.warning(f"Scaler file not found at: {scaler_path}")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(file_path, filename):
    """Convert audio file to WAV format if needed."""
    try:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext != 'wav':
            audio = AudioSegment.from_file(file_path, format=ext)
            wav_path = file_path.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_path, format='wav')
            os.remove(file_path)
            logger.info(f"Converted {filename} to WAV: {wav_path}")
            return wav_path
        return file_path
    except Exception as e:
        logger.error(f"Audio conversion failed for {filename}: {str(e)}")
        return None

def extract_features(file_path):
    """Extract MFCC features from the audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        duration = librosa.get_duration(y=audio, sr=sr)
        if duration < 1.0:
            raise ValueError("Audio duration is too short (< 1 second).")
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        logger.info(f"Extracted features from {file_path}, shape: {mfcc_mean.shape}")
        return mfcc_mean
    except Exception as e:
        logger.error(f"Feature extraction failed for {file_path}: {str(e)}")
        return None

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded audio files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio file upload and predict the language."""
    # Check if a file was uploaded
    if 'file' not in request.files:
        logger.warning("No file part in the request")
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        logger.warning("No file selected")
        return render_template('index.html', error='No selected file')

    # Validate file extension
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file format: {file.filename}")
        return render_template('index.html', error='Invalid file format. Please upload a .wav or .mp3 file')

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    logger.info(f"Saved uploaded file: {file_path}")

    # Convert to WAV if needed
    wav_path = convert_to_wav(file_path, filename)
    if wav_path is None:
        if os.path.exists(file_path):
            os.remove(file_path)
        return render_template('index.html', error='Error converting audio file')

    # Extract features
    features = extract_features(wav_path)
    if features is None:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return render_template('index.html', error='Error processing audio file', filename=filename)

    # Check if required components are loaded
    if label_encoder is None or scaler is None:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        logger.error("Label encoder or scaler unavailable")
        return render_template('index.html', error='Label encoder or scaler unavailable.', filename=filename)

    if not any([rf_model, svm_model, ann_model]):
        if os.path.exists(wav_path):
            os.remove(wav_path)
        logger.error("No models are available for prediction")
        return render_template('index.html', error='No models available for prediction.', filename=filename)

    # Scale features
    try:
        features_scaled = scaler.transform([features])
        logger.info(f"Scaled features shape: {features_scaled.shape}")
    except Exception as e:
        logger.error(f"Feature scaling failed: {str(e)}")
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return render_template('index.html', error=f'Error scaling features: {str(e)}', filename=filename)

    # Make predictions
    predictions = {}
    if rf_model:
        try:
            rf_pred = rf_model.predict(features_scaled)
            rf_label = label_encoder.inverse_transform(rf_pred)[0]
            predictions['Random Forest'] = {'label': rf_label, 'confidence': None}
            logger.info(f"Random Forest predicted: {rf_label} for input shape {features_scaled.shape}")
        except Exception as e:
            logger.error(f"Random Forest prediction failed: {str(e)}")
            predictions['Random Forest'] = {'label': 'Error', 'confidence': None}

    if svm_model:
        try:
            svm_pred = svm_model.predict(features_scaled)
            svm_label = label_encoder.inverse_transform(svm_pred)[0]
            predictions['SVM'] = {'label': svm_label, 'confidence': None}
            logger.info(f"SVM predicted: {svm_label}")
        except Exception as e:
            logger.error(f"SVM prediction failed: {str(e)}")
            predictions['SVM'] = {'label': 'Error', 'confidence': None}

    if ann_model:
        try:
            input_details = ann_model.get_input_details()
            output_details = ann_model.get_output_details()
            ann_model.set_tensor(input_details[0]['index'], features_scaled.astype(np.float32))
            ann_model.invoke()
            ann_pred = ann_model.get_tensor(output_details[0]['index'])
            ann_label = label_encoder.inverse_transform([np.argmax(ann_pred, axis=1)[0]])[0]
            ann_confidence = float(np.max(ann_pred)) * 100
            predictions['ANN'] = {'label': ann_label, 'confidence': ann_confidence}
            logger.info(f"ANN predicted: {ann_label}, confidence: {ann_confidence}%")
        except Exception as e:
            logger.error(f"ANN prediction failed: {str(e)}")
            predictions['ANN'] = {'label': 'Error', 'confidence': None}

    # Keep the file for the audio player; clean up the previous file if it exists
    if not hasattr(app, 'last_uploaded_file'):
        app.last_uploaded_file = None
    if app.last_uploaded_file and os.path.exists(app.last_uploaded_file):
        os.remove(app.last_uploaded_file)
    app.last_uploaded_file = wav_path  # Store the current file path

    return render_template('index.html', predictions=predictions, filename=filename)

@app.route('/manifest.json')
def serve_manifest():
    """Serve the manifest.json file for PWA."""
    return send_from_directory(BASE_DIR, 'manifest.json')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)