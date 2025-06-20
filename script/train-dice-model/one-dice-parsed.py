import os
import json
import shutil
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
TRIAL = 15
NUM_CLASSES = 6
USE_MIXUP = True
USE_CUTMIX = False
ALPHA = 0.2

TRAIN_IMAGES_DIR = 'train/image/one-dice-parsed'
MODEL_SAVE_PATH = f'model/dice-model/one-dice-parsed'
RAPORT_SAVE_PATH = f'raport/train-dice-model/one-dice-parsed'

saved_model = os.path.join(f"model/dice-model/dice-xml-parsed/best-dice-model-trial-4.keras")

os.makedirs(RAPORT_SAVE_PATH, exist_ok=True)
for sub in ['image', 'history', 'class', 'script']:
    os.makedirs(os.path.join(RAPORT_SAVE_PATH, sub), exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    TRAIN_IMAGES_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    TRAIN_IMAGES_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

if os.path.exists(saved_model):
    print(f"Ładowanie istniejącego modelu z: {saved_model}")
    model = tf.keras.models.load_model(saved_model)
else:
    print("Tworzenie nowego modelu od podstaw.")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    for layer in base_model.layers[:-60]:
        layer.trainable = False
    for layer in base_model.layers[-60:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

y_train_labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(class_weights))

checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"best-dice-model-trial-{TRIAL}.keras")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

model.save(os.path.join(MODEL_SAVE_PATH, f"trial-{TRIAL}.keras"))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Dokładność modelu")
plt.legend()
plt.savefig(os.path.join(RAPORT_SAVE_PATH, f'image/trial-{TRIAL}.png'))
plt.clf()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Strata modelu")
plt.legend()
plt.savefig(os.path.join(RAPORT_SAVE_PATH, f'image/loss-trial-{TRIAL}.png'))
plt.clf()

with open(os.path.join(RAPORT_SAVE_PATH, f'history/trial-{TRIAL}.json'), 'w') as f:
    json.dump(history.history, f)

print("Trening zakończony.")
