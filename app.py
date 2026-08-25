from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import librosa
from tensorflow.keras.models import load_model
import sqlite3
from datetime import datetime

app = Flask(__name__)

image_model = load_model('models/image_model.h5')
audio_model = load_model('models/audio_model.h5')
video_model = load_model('models/video_model.h5')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def init_db():
    try:
        conn = sqlite3.connect('history.db')  # Ensure the path is correct
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS checks
                       (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        check_type TEXT,
                        result TEXT,
                        confidence REAL,
                        timestamp TEXT)''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")

def log_check(check_type, result, confidence=None):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("INSERT INTO checks (check_type, result, confidence, timestamp) VALUES (?, ?, ?, ?)",
              (check_type, result, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def extract_audio_features(file_path, max_len=150):
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]
    return mfccs

def extract_video_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(total_frames * i / num_frames) for i in range(num_frames)]
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (128, 128))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return np.array(frames)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-image', methods=['POST'])
def predict_image():
    try:
        file = request.files['image_file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = image_model.predict(img)[0][0]
        result = "Real" if prediction > 0.5 else "Fake"
        confidence = round(float(prediction if prediction > 0.5 else (1 - prediction)) * 100, 2)
        log_check("Image", result, confidence)
        return render_template('result.html', result=result, input_type="Image", detail=file.filename, confidence=confidence)
    except Exception as e:
        print(f"Error: {e}")
        return render_template('error.html')

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    try:
        file = request.files['audio_file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        features = extract_audio_features(filepath)
        features = features[np.newaxis, ..., np.newaxis]

        prediction = audio_model.predict(features)[0][0]
        result = "Real" if prediction > 0.5 else "Fake"
        confidence = round(float(prediction if prediction > 0.5 else (1 - prediction)) * 100, 2)
        log_check("Audio", result, confidence)
        return render_template('result.html', result=result, input_type="Audio", detail=file.filename, confidence=confidence)
    except Exception as e:
        print(f"Error: {e}")
        return render_template('error.html')

@app.route('/predict-video', methods=['POST'])
def predict_video():
    try:
        file = request.files['video_file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        frames = extract_video_frames(filepath) / 255.0
        predictions = video_model.predict(frames)
        avg_prediction = np.mean(predictions)
        result = "Real" if avg_prediction > 0.5 else "Fake"
        confidence = round(float(avg_prediction if avg_prediction > 0.5 else (1 - avg_prediction)) * 100, 2)

        log_check("Video", result, confidence)
        return render_template('result.html', result=result, input_type="Video", detail=file.filename, confidence=confidence)
    except Exception as e:
        print(f"Error: {e}")
        return render_template('error.html')

@app.route('/history')
def history():
    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("SELECT check_type, result, confidence, timestamp FROM checks ORDER BY id DESC LIMIT 20")
        records = c.fetchall()
        conn.close()
        return render_template('history.html', records=records)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return render_template('history.html', records=[])

if __name__ == '__main__':
    try:
        init_db()  # Ensure this line is before app.run()
        app.run(debug=True)
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
