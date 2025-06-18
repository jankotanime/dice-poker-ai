import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import get_file
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

# ======== Ścieżki ========
YOLO_MODEL_PATH = 'model/dice-model/best-train/best-yolo.pt'
IMAGE_PATH = 'app/input-data/image.jpg'
CLASSIFIER_MODEL_PATH = 'model/dice-model/best-train/best-dice-xml.keras'
OUTPUT_IMAGE_PATH = 'app/output-data/image-with-boxes.jpg'

# ======== Ładowanie modeli ========
print("Ładowanie YOLO...")
yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=True)

print("Ładowanie klasyfikatora...")
classifier = load_model(CLASSIFIER_MODEL_PATH)

# Załaduj klasy
with open('app/input-data/class.json', 'r') as f:
    class_names = {int(v): k for k, v in json.load(f).items()}

# ======== Wczytaj obraz ========
image = cv2.imread(IMAGE_PATH)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = yolo(rgb_image)

# ======== Przetwarzanie wyników YOLO ========
predictions = results.pandas().xyxy[0]

for i, row in predictions.iterrows():
    xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
    crop = rgb_image[ymin:ymax, xmin:xmax]

    # Przeskaluj i przetwórz obraz jak do klasyfikatora
    pil_crop = Image.fromarray(crop).resize((224, 224))
    array_crop = img_to_array(pil_crop)
    array_crop = preprocess_input(array_crop)
    array_crop = np.expand_dims(array_crop, axis=0)

    # Klasyfikacja
    pred = classifier.predict(array_crop)
    class_id = np.argmax(pred)
    class_name = class_names[class_id]
    confidence = pred[0][class_id]

    print(f"Obiekt {i+1}: {class_name} ({confidence:.2f})")

    # Dodaj ramkę i etykietę do obrazu
    label = f"{class_name} ({confidence:.2f})"
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(image, label, (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ======== Zapisz wynik ========
cv2.imwrite(OUTPUT_IMAGE_PATH, image)
print(f"Zapisano obraz z ramkami: {OUTPUT_IMAGE_PATH}")
