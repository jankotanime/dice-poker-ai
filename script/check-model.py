import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import torch
from PIL import Image

MODEL_PATH = 'model/dice-model/one-dice-parsed/trial-15.keras'
YOLO_PATH = 'model/dice-model/best-train/best-yolo.pt'
IMAGE_PATH = 'app/input-data/image-3.png'
RAPORT_PATH ='raport/check-model'
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = 'Conv_1'

def get_img_array(pil_img, size):
    pil_img = pil_img.resize(size)
    array = img_to_array(pil_img)
    array = preprocess_input(array)
    return np.expand_dims(array, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), tf.argmax(predictions[0]).numpy(), tf.reduce_max(predictions).numpy()

def display_gradcam(original_img, heatmap, output_path, alpha=0.4):
    img = cv2.resize(original_img, IMG_SIZE)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Oryginalny obraz")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    print("Ładowanie YOLO...")
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_PATH, force_reload=True)

    print(f"Ładowanie modelu z: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Przetwarzanie obrazu: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo(rgb_image, size=1280)
    predictions = results.pandas().xyxy[0]

    print(f"Wykryto {len(predictions)} obiektów.")

    for i, row in predictions.iterrows():
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        crop = rgb_image[ymin:ymax, xmin:xmax]
        pil_crop = Image.fromarray(crop)

        array_crop = get_img_array(pil_crop, IMG_SIZE)

        heatmap, class_id, confidence = make_gradcam_heatmap(array_crop, model, LAST_CONV_LAYER)

        crop_bgr = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
        output_path = os.path.join(RAPORT_PATH, f"object-{i+1}-gradcam.png")
        display_gradcam(crop_bgr, heatmap, output_path)

        print(f" Obiekt {i+1}: klasa {class_id}, pewność {confidence:.2f}")
        print(f" Grad-CAM zapisany do: {output_path}")