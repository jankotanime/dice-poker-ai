import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import json
import shutil
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
TRIAL = 9
NUM_CLASSES = 21

ONE_DICE_MODEL_PATH = 'model/dice-model/best-model/best-one-dice.keras'
DATASET_PATH = 'train/image/two-dices/images'
ANNOTATION_FILE = 'train/image/two-dices/rolls.xml'
MODEL_SAVE_PATH = 'model/dice-model/train/two-dices'
RAPORT_SAVE_PATH = 'raport/train-dice-model/two-dices'

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(RAPORT_SAVE_PATH, 'image'), exist_ok=True)
os.makedirs(os.path.join(RAPORT_SAVE_PATH, 'history'), exist_ok=True)
os.makedirs(os.path.join(RAPORT_SAVE_PATH, 'class'), exist_ok=True)
os.makedirs(os.path.join(RAPORT_SAVE_PATH, 'script'), exist_ok=True)

try:
    current_script = __file__
    script_copy_path = os.path.join(RAPORT_SAVE_PATH, f'script/trial-{TRIAL}.py')
    shutil.copyfile(current_script, script_copy_path)
    print(f'Kopia skryptu zapisana do {script_copy_path}')
except NameError:
    print("Uwaga: nie można zapisać kopii skryptu (__file__ nie istnieje)")

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

class TwoDiceDataGenerator(Sequence):
    def __init__(self, image_label_pairs, batch_size, img_size, num_classes,
                 shuffle=True, augment=False, use_mixup=False, use_cutmix=False, alpha=0.2, mix_strategy='random'):
        self.image_label_pairs = image_label_pairs
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.alpha = alpha
        self.mix_strategy = mix_strategy
        if self.augment:
            self.datagen = train_datagen
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_label_pairs) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_label_pairs)

    def __getitem__(self, index):
        batch_pairs = self.image_label_pairs[index * self.batch_size:(index + 1) * self.batch_size]
        images, labels = [], []
        for img_path, label in batch_pairs:
            img = load_img(img_path, target_size=self.img_size)
            img_array = img_to_array(img)
            if self.augment:
                img_array = self.datagen.random_transform(img_array)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            labels.append(to_categorical(label, num_classes=self.num_classes))

        images = np.array(images)
        labels = np.array(labels)

        if self.augment and (self.use_mixup or self.use_cutmix):
            strategy = self.mix_strategy
            if strategy == 'random':
                strategy = np.random.choice(['mixup', 'cutmix', 'none'])
            if strategy == 'mixup' and self.use_mixup:
                images, labels = self._mixup(images, labels)
            elif strategy == 'cutmix' and self.use_cutmix:
                images, labels = self._cutmix(images, labels)

        return images, labels

    def _mixup(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.shape[0]
        index = np.random.permutation(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y

    def _cutmix(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, h, w, _ = x.shape
        index = np.random.permutation(batch_size)

        rx = np.random.randint(w)
        ry = np.random.randint(h)
        rw = int(w * np.sqrt(1 - lam))
        rh = int(h * np.sqrt(1 - lam))

        x1 = np.clip(rx - rw // 2, 0, w)
        x2 = np.clip(rx + rw // 2, 0, w)
        y1 = np.clip(ry - rh // 2, 0, h)
        y2 = np.clip(ry + rh // 2, 0, h)

        x_cut = x.copy()
        x_cut[:, y1:y2, x1:x2, :] = x[index, y1:y2, x1:x2, :]

        lam_adjusted = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        y_cut = lam_adjusted * y + (1 - lam_adjusted) * y[index]
        return x_cut, y_cut
    
def load_image_label_pairs(xml_path, image_folder):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pairs = []

    def get_symmetric_label(d1, d2):
        d1, d2 = sorted([d1, d2])
        return (d1 - 1) * 6 - (d1 - 1) * d1 // 2 + (d2 - d1)

    for roll in root.findall('roll'):
        filename = roll.find('image').text
        die1 = int(roll.find('die-one').text)
        die2 = int(roll.find('die-two').text)
        full_path = os.path.join(image_folder, filename)
        if os.path.exists(full_path):
            label = get_symmetric_label(die1, die2)
            pairs.append((full_path, label))
    return pairs

image_label_pairs = load_image_label_pairs(ANNOTATION_FILE, DATASET_PATH)

train_pairs, val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=42)
train_generator = TwoDiceDataGenerator(
    train_pairs, BATCH_SIZE, IMG_SIZE, NUM_CLASSES,
    augment=True,
    use_mixup=True,
    use_cutmix=True,
    mix_strategy='random',
)

val_generator = TwoDiceDataGenerator(
    val_pairs, BATCH_SIZE, IMG_SIZE, NUM_CLASSES
)

old_model = tf.keras.models.load_model(ONE_DICE_MODEL_PATH)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

for old_layer in old_model.layers:
    for base_layer in base_model.layers:
        if old_layer.name == base_layer.name:
            try:
                base_layer.set_weights(old_layer.get_weights())
                print(f'Wczytano wagi do warstwy {base_layer.name}')
            except Exception as e:
                print(f'Nie można wczytać wag dla warstwy {base_layer.name}: {e}')
            break

for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"best-dice-model-trial-{TRIAL}.keras")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

model_path = os.path.join(MODEL_SAVE_PATH, f"dice-model-two-dice-trial-{TRIAL}.keras")
model.save(model_path)
print(f'Model zapisany: {model_path}')

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Dokładność modelu")
plt.savefig(os.path.join(RAPORT_SAVE_PATH, f"image/trial-{TRIAL}.png"))
plt.clf()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Strata modelu")
plt.savefig(os.path.join(RAPORT_SAVE_PATH, f"image/loss-trial-{TRIAL}.png"))

with open(os.path.join(RAPORT_SAVE_PATH, f"history/trial-{TRIAL}.json"), "w") as f:
    json.dump(history.history, f)

class_indices = {str(i): i for i in range(NUM_CLASSES)}
with open(os.path.join(RAPORT_SAVE_PATH, f"class/trial-{TRIAL}.json"), "w") as f:
    json.dump(class_indices, f)

print('Skuteczność i klasy zapisane')
