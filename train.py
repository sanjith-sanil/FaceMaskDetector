"""
train.py — Face Mask Detector Training Script
==============================================
Trains a CNN binary classifier to detect face masks.
Saves the trained model weights to 'mymodel.h5'.

Usage:
    python train.py
"""

import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# 1. Build the CNN model
# ─────────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(1, activation='sigmoid'),   # Binary output: mask vs. no mask
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────
# 2. Data generators with augmentation
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'train',                   # Relative path — must be run from project root
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# ─────────────────────────────────────────────
# 3. Train the model  (model.fit replaces the
#    deprecated fit_generator)
# ─────────────────────────────────────────────
history = model.fit(
    training_set,
    epochs=10,
    validation_data=test_set,
)

# ─────────────────────────────────────────────
# 4. Save weights
# ─────────────────────────────────────────────
model.save('mymodel.h5')
print("\n✅ Model saved to mymodel.h5")
print(f"   Training accuracy  : {history.history['accuracy'][-1]*100:.2f}%")
print(f"   Validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
