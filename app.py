import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Suppress GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import numpy as np
import librosa
import pickle
import tensorflow as tf  # For TFLite
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from pydub import AudioSegment

app = Flask(__name__)
app.static_folder = 'static'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3'}  # Updated to support only .wav and .mp3
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models, label encoder, and scaler
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
        print(f"Random Forest model loaded successfully from {rf_path}")
    except Exception as e:
        print(f"Error loading Random Forest model from {rf_path}: {str(e)}")
else:
    print(f"Random Forest model file not found at: {rf_path}")

# Load SVM model
svm_path = os.path.join(models_dir, 'svm_model.pkl')
if os.path.exists(svm_path):
    try:
        with open(svm_path, 'rb') as f:
            svm_model = pickle.load(f)
        print("SVM model loaded successfully")
    except Exception as e:
        print(f"Error loading SVM model: {e}")
else:
    print(f"SVM model file not found at: {svm_path}")

# Load ANN model as TFLite
ann_path = os.path.join(models_dir, 'ann_model.tflite')
if os.path.exists(ann_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=ann_path)
        interpreter.allocate_tensors()
        ann_model = interpreter
        print("TFLite ANN model loaded successfully")
    except Exception as e:
        print(f"Error loading TFLite ANN model: {e}")
        ann_model = None
else:
    print(f"TFLite ANN model file not found at: {ann_path}")
    ann_model = None

# Load label encoder
label_encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
if os.path.exists(label_encoder_path):
    try:
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Label encoder loaded successfully")
    except Exception as e:
        print(f"Error loading label encoder: {e}")
else:
    print(f"Label encoder file not found at: {label_encoder_path}")

# Load scaler
scaler_path = os.path.join(models_dir, 'scaler.pkl')
if os.path.exists(scaler_path):
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully")
    except Exception as e:
        print(f"Error loading scaler: {e}")
else:
    print(f"Scaler file not found at: {scaler_path}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(file_path, filename):
    try:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext != 'wav':
            audio = AudioSegment.from_file(file_path, format=ext)
            wav_path = file_path.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_path, format='wav')
            os.remove(file_path)
            return wav_path
        return file_path
    except Exception as e:
        print(f"Audio conversion failed: {e}")
        return None

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        duration = librosa.get_duration(y=audio, sr=sr)
        if duration < 1.0:
            raise ValueError("Audio duration is too short (< 1 second).")
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        print(f"Extracted features shape: {mfcc_mean.shape}")
        return mfcc_mean
    except Exception as e:
        print(f"Feature extraction failed for {file_path}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        wav_path = convert_to_wav(file_path, filename)
        if wav_path is None:
            if os.path.exists(file_path):
                os.remove(file_path)
            return render_template('index.html', error='Error converting audio file')
        features = extract_features(wav_path)
        if features is None:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return render_template('index.html', error='Error processing audio file', filename=filename)
        if label_encoder is None or scaler is None or (rf_model is None and svm_model is None and ann_model is None):
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return render_template('index.html', error='Models or label encoder unavailable.', filename=filename)
        try:
            features_scaled = scaler.transform([features])
            print(f"Scaled features shape: {features_scaled.shape}")
        except Exception as e:
            print(f"Feature scaling failed: {e}")
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return render_template('index.html', error=f'Error scaling features: {str(e)}', filename=filename)
        predictions = {}
        if rf_model:
            try:
                rf_pred = rf_model.predict(features_scaled)
                rf_label = label_encoder.inverse_transform(rf_pred)[0]
                predictions['Random Forest'] = {'label': rf_label, 'confidence': None}
                print(f"Random Forest predicted: {rf_label} for input shape {features_scaled.shape}")
            except Exception as e:
                print(f"Random Forest prediction failed: {str(e)}")
        else:
            print("Random Forest model is not loaded")
        if svm_model:
            try:
                svm_pred = svm_model.predict(features_scaled)
                svm_label = label_encoder.inverse_transform(svm_pred)[0]
                predictions['SVM'] = {'label': svm_label, 'confidence': None}
                print(f"SVM predicted: {svm_label}")
            except Exception as e:
                print(f"SVM prediction failed: {e}")
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
                print(f"ANN predicted: {ann_label}, confidence: {ann_confidence}%")
            except Exception as e:
                print(f"ANN prediction failed: {e}")
        # Keep the file for the audio player; clean up the previous file if it exists
        if not hasattr(app, 'last_uploaded_file'):
            app.last_uploaded_file = None
        if app.last_uploaded_file and os.path.exists(app.last_uploaded_file):
            os.remove(app.last_uploaded_file)
        app.last_uploaded_file = wav_path  # Store the current file path
        return render_template('index.html', predictions=predictions, filename=filename)
    else:
        return render_template('index.html', error='Invalid file format. Please upload a .wav or .mp3 file')  # Updated error message

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory(BASE_DIR, 'manifest.json')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
