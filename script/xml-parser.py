import os
import cv2
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

# === Ścieżki do dostosowania ===
YOLO_MODEL_PATH = 'model/dice-model/best-train/best-yolo.pt'  # <- model YOLO
IMAGES_DIR = 'train/image/random-dices-xml'       # <- folder z obrazkami
ANNOTATIONS_DIR = 'train/image/random-dices-xml'     # <- folder z plikami XML
OUTPUT_DIR = 'train/image/dice-xml-parsed' # <- folder docelowy na przycięte obrazki
# ===============================

# Ładowanie YOLO
print("[INFO] Ładowanie YOLO...")
yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=True)

# Pomocnicza funkcja do wczytania etykiety z XML na podstawie pliku
def get_label_from_xml(filename):
    xml_file = os.path.splitext(filename)[0] + '.xml'
    xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
    if not os.path.exists(xml_path):
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text.lower().strip()
        if 'datos' in name:
            return None
        if 'cara' in name:
            try:
                return str(int(name.replace('cara', '').strip()))
            except:
                return None
    return None

# Tworzymy foldery wyjściowe
for i in range(1, 7):
    os.makedirs(os.path.join(OUTPUT_DIR, str(i)), exist_ok=True)

# Przetwarzanie obrazów
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in tqdm(image_files, desc="Przetwarzanie obrazów"):
    image_path = os.path.join(IMAGES_DIR, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] Nie można wczytać: {image_path}")
        continue

    # Pobierz etykietę z XML
    class_label = get_label_from_xml(filename)
    if class_label is None:
        print(f"[INFO] Pominięto (brak etykiety): {filename}")
        continue

    # YOLO przycina
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
