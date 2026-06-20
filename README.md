# Face Mask Detector

A real-time face mask detection app powered by deep learning and a webcam. It uses a trained Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask or not — live, frame by frame. Multiple faces can be detected simultaneously, each labeled with a confidence score.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Deep Learning | TensorFlow 2.21, Keras 3 |
| Computer Vision | OpenCV 4.13 |
| Face Detection | Haar Cascade Classifier |
| Model Format | HDF5 (`.h5`) |

---

## Prerequisites

- **Python 3.12** installed → [Download](https://www.python.org/downloads/)
- A **webcam** connected to your computer
- **Git** (optional, for cloning) → [Download](https://git-scm.com/)

---

## Installation & Setup

### 1. Clone or Download the Project

```bash
git clone https://github.com/your-username/FaceMaskDetector.git
cd FaceMaskDetector
```

Or download and extract the ZIP, then open a terminal in the project folder.

### 2. Create a Virtual Environment

```bash
# Create the environment
python -m venv venv312

# Activate it — Windows
venv312\Scripts\activate

# Activate it — macOS / Linux
source venv312/bin/activate
```

### 3. Install Dependencies

```bash
pip install tensorflow opencv-python h5py numpy
```

> No environment variables are required for this project.

---

## Running the Project

> **If `mymodel.h5` already exists in the folder, skip straight to Step 2.**

### Step 1 — Train the Model (first time only)

```bash
python train.py
```

This trains the CNN for 10 epochs and saves the weights to `mymodel.h5`. Takes a few minutes depending on your hardware.

### Step 2 — Start Live Detection

```bash
python detect.py
```

A webcam window opens. Each face detected is boxed with a label (`MASK` or `NO MASK`) and a confidence percentage. Press **`q`** to quit.

### Step 2 (Alternative) — Test on a Single Image

```bash
python detect.py --image path/to/your/image.jpg
```

Example:

```bash
python detect.py --image test/with_mask/1-with-mask.jpg
```

### Quick Launch (Windows only)

Double-click `run_detector.bat` — it activates the environment and starts the detector automatically.

---

## Project Structure

```
FaceMaskDetector/
│
├── train.py                          # Trains the CNN and saves mymodel.h5
├── detect.py                         # Runs live or single-image detection
├── run_detector.bat                  # One-click launcher (Windows)
│
├── mymodel.h5                        # Pre-trained model weights
├── haarcascade_frontalface_default.xml  # Face detection classifier
├── requirements.txt                  # Full dependency list (pinned versions)
│
├── train/                            # Training images
│   ├── with_mask/
│   └── without_mask/
│
└── test/                             # Test images
    ├── with_mask/
    └── without_mask/
```

---

## How It Works

1. **Face Localisation** — OpenCV's Haar Cascade scans each video frame for faces.
2. **Preprocessing** — Each detected face is resized to 150×150 px and normalised in memory (no temporary files written to disk).
3. **Classification** — The CNN outputs a sigmoid score; values above 0.5 → No Mask, below → Mask.
4. **Display** — A colour-coded bounding box and confidence score are drawn on the frame in real time.

---

## Model Performance

| Dataset | Accuracy |
|---|---|
| Training set | 98.2% |
| Test set | 97.3% |

---

## Dataset

1,314 training images and 194 test images across two classes (`with_mask` / `without_mask`).

Download: [Data Flair Face Mask Dataset](https://data-flair.training/blogs/download-face-mask-data/)
