import os
import xml.etree.ElementTree as ET

dataset_dir = 'train/image/random-dices-xml/'
labels_dir = os.path.join(dataset_dir, 'labels')
os.makedirs(labels_dir, exist_ok=True)

class_map = {
    'Cara 1': 0,
    'Cara 2': 1,
    'Cara 3': 2,
    'Cara 4': 3,
    'Cara 5': 4,
    'Cara 6': 5
}

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return x_center * dw, y_center * dh, w * dw, h * dh

for xml_file in os.listdir(dataset_dir):
    if not xml_file.endswith('.xml'):
        continue

    xml_path = os.path.join(dataset_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = root.findall('object')
    if not objects:
        continue

    txt_filename = xml_file.replace('.xml', '.txt')
    txt_path = os.path.join(labels_dir, txt_filename)

    with open(txt_path, 'w') as out_file:
        for obj in objects:
            cls_name = obj.find('name').text
            if cls_name not in class_map:
                continue
            cls_id = class_map[cls_name]
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            bbox_converted = convert_bbox((width, height), (xmin, ymin, xmax, ymax))
            out_file.write(f"{cls_id} {' '.join(map(str, bbox_converted))}\n")

print("Konwersja XML do YOLO gotowa!")
