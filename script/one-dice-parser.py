import os
import cv2
import torch
from PIL import Image
from tqdm import tqdm

YOLO_MODEL_PATH = 'model/dice-model/best-train/best-yolo.pt'
IMAGES_ROOT_DIR = 'train/image/one-dice'
OUTPUT_DIR = 'train/image/one-dice-parsed'

print("[INFO] Ładowanie YOLO...")
yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=True)

for i in range(1, 7):
    os.makedirs(os.path.join(OUTPUT_DIR, str(i)), exist_ok=True)

for subdir, dirs, files in os.walk(IMAGES_ROOT_DIR):
    for filename in tqdm(files, desc=f"Przetwarzanie {subdir}"):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(subdir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Nie można wczytać: {image_path}")
            continue

        folder_name = os.path.basename(subdir).lower()
        if not folder_name.startswith('dice-'):
            print(f"[INFO] Pominięto (niewłaściwy folder): {folder_name}")
            continue

        try:
            class_label = str(int(folder_name.replace('dice-', '').strip()))
        except ValueError:
            print(f"[WARN] Niepoprawna etykieta w folderze: {folder_name}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = yolo(rgb_image, size=1280)
        detections = results.pandas().xyxy[0]

        for i, row in detections.iterrows():
            xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            crop = image[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_filename = f"{os.path.splitext(filename)[0]}_{i}.jpg"
            output_path = os.path.join(OUTPUT_DIR, class_label, crop_filename)
            crop_pil.save(output_path)

            print(f"[OK] Zapisano: {output_path}")
