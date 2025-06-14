import os
os.system('python3 yolov5/train.py --img 416 --batch 2 --epochs 100 --data script/dice-yolo.yaml --device cpu')
