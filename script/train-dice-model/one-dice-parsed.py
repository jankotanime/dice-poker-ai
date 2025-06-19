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

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
TRIAL = 3
NUM_CLASSES = 6
USE_MIXUP = True
USE_CUTMIX = False
ALPHA = 0.2

TRAIN_IMAGES_DIR = 'train/image/one-dice-parsed'
MODEL_SAVE_PATH = f'model/dice-model/one-dice-parsed'
RAPORT_SAVE_PATH = f'raport/train-dice-model/one-dice-parsed'

saved_model = os.path.join(f"model/dice-model/best-train/best-dice-xml.keras")


os.makedirs(RAPORT_SAVE_PATH, exist_ok=True)
for sub in ['image', 'history', 'class', 'script']:
    os.makedirs(os.path.join(RAPORT_SAVE_PATH, sub), exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

class_names = sorted([d for d in os.listdir(TRAIN_IMAGES_DIR) if os.path.isdir(os.path.join(TRAIN_IMAGES_DIR, d))])
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

image_paths = []
labels = []

for cls in class_names:
    cls_folder = os.path.join(TRAIN_IMAGES_DIR, cls)
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for path in glob(os.path.join(cls_folder, ext)):
            image_paths.append(path)
            labels.append(class_to_idx[cls])

image_paths = np.array(image_paths)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)

class DiceDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, img_size, num_classes, 
                 shuffle=True, augment=True, mixup=False, cutmix=False, alpha=0.2):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.mixup = mixup
        self.cutmix = cutmix
        self.alpha = alpha
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        idxs = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = [self.image_paths[i] for i in idxs]
        batch_y = [self.labels[i] for i in idxs]

        images = np.array([preprocess_input(img_to_array(load_img(p, target_size=self.img_size))) for p in batch_x])
        labels = to_categorical(batch_y, num_classes=self.num_classes)

        if self.mixup:
            return self.apply_mixup(images, labels)
        elif self.cutmix:
            return self.apply_cutmix(images, labels)
        else:
            return images, labels

    def apply_mixup(self, images, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        indices = np.random.permutation(len(images))
        x1, x2 = images, images[indices]
        y1, y2 = labels, labels[indices]
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        return mixed_x, mixed_y

    def apply_cutmix(self, images, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        indices = np.random.permutation(len(images))
        x1, x2 = images, images[indices]
        y1, y2 = labels, labels[indices]

        h, w = self.img_size
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        cut_w = int(w * np.sqrt(1 - lam))
        cut_h = int(h * np.sqrt(1 - lam))
        x1_1 = np.clip(cx - cut_w // 2, 0, w)
        y1_1 = np.clip(cy - cut_h // 2, 0, h)
        x2_1 = np.clip(cx + cut_w // 2, 0, w)
        y2_1 = np.clip(cy + cut_h // 2, 0, h)

        x1[:, y1_1:y2_1, x1_1:x2_1, :] = x2[:, y1_1:y2_1, x1_1:x2_1, :]
        lam = 1 - ((x2_1 - x1_1) * (y2_1 - y1_1) / (w * h))
        mixed_y = lam * y1 + (1 - lam) * y2

        return x1, mixed_y

train_gen = DiceDataGenerator(X_train, y_train, BATCH_SIZE, IMG_SIZE, NUM_CLASSES, mixup=USE_MIXUP, cutmix=USE_CUTMIX, alpha=ALPHA)
val_gen = DiceDataGenerator(X_val, y_val, BATCH_SIZE, IMG_SIZE, NUM_CLASSES, shuffle=False, augment=False)

checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"best-dice-model-trial-{TRIAL}.keras")

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
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.002))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

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

with open(os.path.join(RAPORT_SAVE_PATH, f'class/trial-{TRIAL}.json'), 'w') as f:
    json.dump(class_to_idx, f)

try:
    shutil.copy(__file__, os.path.join(RAPORT_SAVE_PATH, f'script/trial-{TRIAL}.py'))
except NameError:
    pass

print("Trening zakończony.")
