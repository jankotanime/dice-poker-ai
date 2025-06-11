import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import json
import shutil

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
DATASET_PATH = 'train/image/one-dice'
MODEL_SAVE_PATH = 'model/train-dice-model/one-dice'
RAPORT_SAVE_PATH = 'raport/train-dice-model/one-dice'
TRIAL = 6

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

model.save(os.path.join(MODEL_SAVE_PATH, "dice-model-one-dice-trial-"+str(TRIAL)+".keras"))
print(f'Model zapisany')

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Dokładność modelu")
plt.savefig(os.path.join(RAPORT_SAVE_PATH, "image/trial-"+str(TRIAL)+".png"))
with open(os.path.join(RAPORT_SAVE_PATH, "history/trial-"+str(TRIAL)+".json"), "w") as f:
    json.dump(history.history, f)
print(f'Skuteczność skryptu zapisana')

try:
    current_script = __file__
    script_copy_path = os.path.join(RAPORT_SAVE_PATH, 'script/trial-'+str(TRIAL)+'.py')
    shutil.copyfile(current_script, script_copy_path)
    print(f'Kopia skryptu zapisana')
except NameError:
    print("Uwaga: nie można zapisać kopii skryptu")