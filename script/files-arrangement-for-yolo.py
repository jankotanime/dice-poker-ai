import os
import shutil
import random

images_dir = 'train/image/one-dice-yolo/images'
labels_dir = 'train/image/one-dice-yolo/labels'

train_img_dir = 'train/image/one-dice-yolo/images/train'
val_img_dir = 'train/image/one-dice-yolo/images/val'
train_label_dir = 'train/image/one-dice-yolo/labels/train'
val_label_dir = 'train/image/one-dice-yolo/labels/val'

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
random.shuffle(all_images)

split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

for img_name in train_images:
    shutil.move(os.path.join(images_dir, img_name), train_img_dir)
    label_name = os.path.splitext(img_name)[0] + '.txt'
    shutil.move(os.path.join(labels_dir, label_name), train_label_dir)

for img_name in val_images:
    shutil.move(os.path.join(images_dir, img_name), val_img_dir)
    label_name = os.path.splitext(img_name)[0] + '.txt'
    shutil.move(os.path.join(labels_dir, label_name), val_label_dir)

print('Podział datasetu wykonany')
