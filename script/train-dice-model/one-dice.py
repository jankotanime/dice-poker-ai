import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import json
import shutil
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_model(filters=[32, 64, 128], dropout=0.3):
    model = models.Sequential()
    model.add(layers.Input(shape=(*IMG_SIZE, 3)))
    for f in filters:
        model.add(layers.Conv2D(f, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(6, activation='softmax'))
    return model

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = 'train/image/one-dice'
MODEL_SAVE_PATH = 'model/dice-model/one-dice'
RAPORT_SAVE_PATH = 'raport/train-dice-model/one-dice'
TRIAL = 14

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

model = create_model(filters=[32, 64], dropout=0.3)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"best-dice-model-trial-{TRIAL}.keras")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

model.save(os.path.join(MODEL_SAVE_PATH, "trial-"+str(TRIAL)+".keras"))

print(f'Model zapisany')

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Dokładność modelu")
plt.savefig(os.path.join(RAPORT_SAVE_PATH, "image/trial-"+str(TRIAL)+".png"))

plt.clf()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Strata modelu")
plt.savefig(os.path.join(RAPORT_SAVE_PATH, "image/loss-trial-"+str(TRIAL)+".png"))

with open(os.path.join(RAPORT_SAVE_PATH, "history/trial-"+str(TRIAL)+".json"), "w") as f:
    json.dump(history.history, f)

with open(os.path.join(RAPORT_SAVE_PATH, f"class/trial-{TRIAL}.json"), "w") as f:
    json.dump(train_generator.class_indices, f)

print(f'Skuteczność skryptu zapisana')

try:
    current_script = __file__
    script_copy_path = os.path.join(RAPORT_SAVE_PATH, 'script/trial-'+str(TRIAL)+'.py')
    shutil.copyfile(current_script, script_copy_path)
    print(f'Kopia skryptu zapisana')
except NameError:
    print("Uwaga: nie można zapisać kopii skryptu")