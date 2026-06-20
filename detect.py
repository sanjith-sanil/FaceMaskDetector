"""
detect.py — Face Mask Detector Inference Script
================================================
Performs real-time face mask detection using a webcam,
or runs inference on a single image file.

Usage:
    # Live webcam detection:
    python detect.py

    # Single image test:
    python detect.py --image path/to/image.jpg
"""

import argparse
import datetime
import os

# Suppress verbose TensorFlow info/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_PATH       = 'mymodel.h5'
CASCADE_PATH     = 'haarcascade_frontalface_default.xml'
INPUT_SIZE       = (150, 150)
MASK_COLOR       = (0, 255,   0)   # Green  — mask detected
NO_MASK_COLOR    = (0,   0, 255)   # Red    — no mask
CONFIDENCE_THRESH = 0.5            # Sigmoid threshold


def preprocess_face(face_roi: np.ndarray) -> np.ndarray:
    """
    Resize and normalise a BGR face crop for model input.
    Returns a (1, 150, 150, 3) float32 array — no disk I/O required.
    """
    resized = cv2.resize(face_roi, INPUT_SIZE)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr     = rgb.astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)


def predict_mask(model, face_roi: np.ndarray):
    """
    Returns (label, confidence, color) for a face crop.
    Uses pred > 0.5 instead of pred == 1 (sigmoid is a float).
    """
    tensor = preprocess_face(face_roi)
    pred   = model.predict(tensor, verbose=0)[0][0]

    if pred > CONFIDENCE_THRESH:
        label      = 'NO MASK'
        confidence = pred
        color      = NO_MASK_COLOR
    else:
        label      = 'MASK'
        confidence = 1.0 - pred
        color      = MASK_COLOR

    return label, confidence, color


def draw_annotation(frame, x, y, w, h, label, confidence, color):
    """Draw bounding box, label, and confidence score on the frame."""
    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    # Label + confidence percentage
    display_text = f'{label} ({confidence * 100:.1f}%)'
    text_x       = max(x, 0)
    text_y       = max(y - 10, 20)
    cv2.putText(frame, display_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# ─────────────────────────────────────────────
# Single-image inference
# ─────────────────────────────────────────────
def run_single_image(model, image_path: str):
    """Run prediction on a single image file and display the result."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces        = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        print("[WARNING] No faces detected in the image.")
    else:
        for (x, y, w, h) in faces:
            face_roi                  = frame[y:y + h, x:x + w]
            label, confidence, color  = predict_mask(model, face_roi)
            draw_annotation(frame, x, y, w, h, label, confidence, color)
            print(f"   → {label}  ({confidence * 100:.1f}% confidence)")

    cv2.imshow('Face Mask Detection — Single Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Live webcam detection
# ─────────────────────────────────────────────
def run_live(model):
    """Perform real-time face mask detection using the default webcam."""
    cap          = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[OK] Live detection started. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            face_roi                  = frame[y:y + h, x:x + w]
            label, confidence, color  = predict_mask(model, face_roi)
            draw_annotation(frame, x, y, w, h, label, confidence, color)

        # Timestamp overlay
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Face Mask Detection — Live', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[STOPPED] Detection stopped.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Mask Detector')
    parser.add_argument(
        '--image', type=str, default=None,
        help='Path to a single image file for inference. '
             'Omit to use live webcam detection.'
    )
    args = parser.parse_args()

    print(f"[...] Loading model from '{MODEL_PATH}' ...")
    mymodel = load_model(MODEL_PATH)
    print("[OK] Model loaded.\n")

    if args.image:
        run_single_image(mymodel, args.image)
    else:
        run_live(mymodel)
