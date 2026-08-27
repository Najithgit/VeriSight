import os

# Reduce TensorFlow CPU/thread usage on small cloud instances
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

# Prevent Numba/LLVM JIT compilation on Render
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_DISABLE_COVERAGE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import gc
import sqlite3
from datetime import datetime

import numpy as np
import cv2
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model


app = Flask(__name__)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

UPLOAD_FOLDER = "uploads"
DATABASE = "history.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# TensorFlow configuration
# ---------------------------------------------------------

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass


# ---------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------

_image_model = None
_audio_model = None
_video_model = None


def get_image_model():
    global _image_model

    if _image_model is None:
        print("Loading image model...")
        _image_model = load_model("models/image_model.h5")
        print("Image model loaded.")

    return _image_model


def get_audio_model():
    global _audio_model

    if _audio_model is None:
        print("Loading audio model...")
        _audio_model = load_model("models/audio_model.h5")
        print("Audio model loaded.")

    return _audio_model


def get_video_model():
    global _video_model

    if _video_model is None:
        print("Loading video model...")
        _video_model = load_model("models/video_model.h5")
        print("Video model loaded.")

    return _video_model


# ---------------------------------------------------------
# Database
# ---------------------------------------------------------

def init_db():
    try:
        conn = sqlite3.connect(DATABASE)

        c = conn.cursor()

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_type TEXT,
                result TEXT,
                confidence REAL,
                timestamp TEXT
            )
            """
        )

        conn.commit()
        conn.close()

    except sqlite3.Error as e:
        print(f"Database connection error: {e}")


def log_check(check_type, result, confidence=None):
    conn = sqlite3.connect(DATABASE)

    c = conn.cursor()

    c.execute(
        """
        INSERT INTO checks
        (check_type, result, confidence, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (
            check_type,
            result,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )

    conn.commit()
    conn.close()


# Initialize database when Gunicorn imports app
init_db()


# ---------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------

def extract_audio_features(file_path, max_len=150):

    audio, sr = librosa.load(
        file_path,
        sr=16000
    )

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    if mfccs.shape[1] < max_len:

        pad_width = max_len - mfccs.shape[1]

        mfccs = np.pad(
            mfccs,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:

        mfccs = mfccs[:, :max_len]

    return mfccs


# ---------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------

def extract_video_frames(video_path, num_frames=10):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise ValueError("Unable to read video frames.")

    frame_indices = [
        int(total_frames * i / num_frames)
        for i in range(num_frames)
    ]

    frames = []

    for idx in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            idx
        )

        ret, frame = cap.read()

        if ret:

            frame = cv2.resize(
                frame,
                (128, 128)
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    return np.array(frames)


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------
# IMAGE PREDICTION
# ---------------------------------------------------------

@app.route("/predict-image", methods=["POST"])
def predict_image():

    filepath = None

    try:

        file = request.files["image_file"]

        if not file or file.filename == "":
            raise ValueError("No image file selected.")

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        # Load model only when image detection is requested
        image_model = get_image_model()

        img = cv2.imread(filepath)

        if img is None:
            raise ValueError("Unable to read image.")

        img = cv2.resize(
            img,
            (128, 128)
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = img / 255.0

        img = np.expand_dims(
            img,
            axis=0
        )

        prediction = image_model.predict(
            img,
            verbose=0
        )[0][0]

        result = (
            "Real"
            if prediction > 0.5
            else "Fake"
        )

        confidence = round(
            float(
                prediction
                if prediction > 0.5
                else (1 - prediction)
            ) * 100,
            2
        )

        log_check(
            "Image",
            result,
            confidence
        )

        return render_template(
            "result.html",
            result=result,
            input_type="Image",
            detail=file.filename,
            confidence=confidence
        )

    except Exception as e:

        print(f"Image prediction error: {e}")

        return f"Image prediction failed: {str(e)}", 500
    finally:

        if filepath and os.path.exists(filepath):

            try:
                os.remove(filepath)
            except Exception:
                pass

        gc.collect()


# ---------------------------------------------------------
# AUDIO PREDICTION
# ---------------------------------------------------------

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    try:
        print("AUDIO: request received", flush=True)

        file = request.files['audio_file']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        print("AUDIO: file saved", flush=True)

        print("AUDIO: starting feature extraction", flush=True)
        features = extract_audio_features(filepath)
        print(f"AUDIO: features extracted, shape={features.shape}", flush=True)

        features = features[np.newaxis, ..., np.newaxis]
        print(f"AUDIO: final input shape={features.shape}", flush=True)

        print("AUDIO: starting model prediction", flush=True)
        prediction = audio_model.predict(features, verbose=0)[0][0]
        print(f"AUDIO: prediction completed = {prediction}", flush=True)

        result = "Real" if prediction > 0.5 else "Fake"
        confidence = round(
            float(prediction if prediction > 0.5 else (1 - prediction)) * 100,
            2
        )

        log_check("Audio", result, confidence)

        print("AUDIO: result generated successfully", flush=True)

        return render_template(
            'result.html',
            result=result,
            input_type="Audio",
            detail=file.filename,
            confidence=confidence
        )

    except Exception as e:
        print(f"AUDIO ERROR: {e}", flush=True)
        return render_template('error.html')


# ---------------------------------------------------------
# VIDEO PREDICTION
# ---------------------------------------------------------

@app.route("/predict-video", methods=["POST"])
def predict_video():

    filepath = None

    try:

        file = request.files["video_file"]

        if not file or file.filename == "":
            raise ValueError("No video file selected.")

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        frames = extract_video_frames(
            filepath,
            num_frames=10
        )

        frames = frames / 255.0

        # Load model only when video detection is requested
        video_model = get_video_model()

        predictions = video_model.predict(
            frames,
            verbose=0
        )

        avg_prediction = np.mean(
            predictions
        )

        result = (
            "Real"
            if avg_prediction > 0.5
            else "Fake"
        )

        confidence = round(
            float(
                avg_prediction
                if avg_prediction > 0.5
                else (1 - avg_prediction)
            ) * 100,
            2
        )

        log_check(
            "Video",
            result,
            confidence
        )

        return render_template(
            "result.html",
            result=result,
            input_type="Video",
            detail=file.filename,
            confidence=confidence
        )

    except Exception as e:

        print(f"Video prediction error: {e}")

        return f"Video prediction failed: {str(e)}", 500

    finally:

        if filepath and os.path.exists(filepath):

            try:
                os.remove(filepath)
            except Exception:
                pass

        gc.collect()


# ---------------------------------------------------------
# HISTORY
# ---------------------------------------------------------

@app.route("/history")
def history():

    try:

        conn = sqlite3.connect(
            DATABASE
        )

        c = conn.cursor()

        c.execute(
            """
            SELECT
                check_type,
                result,
                confidence,
                timestamp
            FROM checks
            ORDER BY id DESC
            LIMIT 20
            """
        )

        records = c.fetchall()

        conn.close()

        return render_template(
            "history.html",
            records=records
        )

    except sqlite3.Error as e:

        print(
            f"Database error: {e}"
        )

        return render_template(
            "history.html",
            records=[]
        )


# ---------------------------------------------------------
# Local development
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=True
    )