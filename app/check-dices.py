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

TRIAL = 7

YOLO_MODEL_PATH = 'model/dice-model/best-train/best-yolo.pt'
IMAGE_PATH = f'app/input-data/image-{TRIAL}.png'
CLASSIFIER_MODEL_PATH = 'model/dice-model/one-dice-parsed/best-dice-model-trial-14.keras'
OUTPUT_IMAGE_PATH = f'app/output-data/image-with-boxes-{TRIAL}.jpg'

print("Ładowanie YOLO...")
yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=True)

print("Ładowanie klasyfikatora...")
classifier = load_model(CLASSIFIER_MODEL_PATH)

with open('app/input-data/class.json', 'r') as f:
    class_names = {int(v): k for k, v in json.load(f).items()}

image = cv2.imread(IMAGE_PATH)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = yolo(rgb_image, size=1280)

predictions = results.pandas().xyxy[0]

def is_near(box1, box2, threshold=10):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    return (abs(x1 - a1) < threshold and abs(y1 - b1) < threshold and
            abs(x2 - a2) < threshold and abs(y2 - b2) < threshold)

def count_dots(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height = crop.shape[0]
    max_diameter = 0.2 * height

    dots = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        ratio = w / h
        if ratio < 0.75 or ratio > 1.25:
            continue

        diameter = max(w, h)
        if diameter > max_diameter:
            continue

        if cv2.contourArea(cnt) < 25:
            continue

        dots.append(cnt)

    return len(dots)



filtered_boxes = []
final_predictions = []

for i, row in predictions.iterrows():
    box = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))

    if any(is_near(box, prev_box) for prev_box in filtered_boxes):
        continue

    filtered_boxes.append(box)
    final_predictions.append(row)

print(f"Znaleziono kości po filtrze: {len(final_predictions)}")

for row in final_predictions:
    xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
    crop = rgb_image[ymin:ymax, xmin:xmax]
    dot_count = count_dots(crop)

    yolo_class_name = row['name']
    yolo_conf = row['confidence']

    pil_crop = Image.fromarray(crop).resize((224, 224))
    array_crop = img_to_array(pil_crop)
    array_crop = preprocess_input(array_crop)
    array_crop = np.expand_dims(array_crop, axis=0)

    pred = classifier.predict(array_crop)
    
    top2_indices = pred[0].argsort()[-2:][::-1]
    best_id, second_id = top2_indices
    best_confidence = pred[0][best_id]

    class_name = class_names[best_id]
    dots = dot_count

    if best_id+1 == dots:
        final_class = best_id+1
    elif 0 < dots < 5:
        final_class = dots
    elif best_id == 4:
        if best_confidence > 0.995 or dots == 5:
            final_class = 5
        else:
            final_class = 6
    else:
        final_class = best_id+1

    print(f"YOLO: {yolo_class_name} ({yolo_conf:.2f})")
    print(f"Klasyfikator: {class_name} ({pred[0][best_id]:.2f})")
    print("Ilość oczek:", dot_count)
    print("Ostateczny wynik:", final_class)

    label = f"Liczba oczek: {final_class}"

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(image, label, (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite(OUTPUT_IMAGE_PATH, image)
print(f"Zapisano obraz z ramkami: {OUTPUT_IMAGE_PATH}")
