# Dice Poker AI
### Version 1.0
---
Dice Poker AI is an experimental project developed for the Computational Intelligence course. It integrates YOLO, a neural network, and a decision tree. It also includes a report in Polish, algorithms for evaluating game profitability, and a terminal application that lets you play dice poker with an AI by sending it images. [Go to the report](https://github.com/jankotanime/dice-poker-ai/blob/main/raport/raport.md)

## Runtime environment
### Requirements
- Python
- pip packages: tensorflow, torch, numpy, opencv-python

## How to run
1. Make sure you have added images named `image-{number}.png` in the `app/input-data/` directory. `Example: app/input-data/image-4.png`
2. For bash script: run the app using `./app-run.sh`
3. Alternatively, run the app with `python3 app/main.py` (for older Python versions: `python app/main.py`)

## Features
- **Dice recognition** – The model classifies dice values based on the image, using a hybrid of machine learning (MobileNetV2 in .keras format) and image processing (OpenCV). Recognition accuracy reaches around 80% in app tests.

- **Dice localization** – YOLO identifies the position of dice in the images with high precision (mAP@0.5: 98%).

- **Game strategy evaluation** – An analytical script assesses the optimal dice to re-roll, balancing risk and maximizing expected points.

- **Betting decisions** – A decision tree evaluates whether it is profitable to raise the bet, visualizing how different features influence the decision.

- **Console application** – Enables playing against a bot that makes decisions based on trained models.

- **Grad-CAM visualization** – A tool for model error analysis, showing which parts of the image were most influential in the model’s decision.

## Infrastructure and accuracy

| Function               | Model                    | Accuracy       |
| ---------------------- | ------------------------ | -------------- |
| Dice recognition        | MobileNetV2 + OpenCV      | ~80%           |
| Dice localization       | YOLOv5                    | mAP@0.5: 98%   |
| Re-roll decision        | Heuristic algorithm       | EV: +2.45 pts  |
| Betting decision        | Decision tree             | 94% test acc   |

The project is implemented in Python.

## Preview
### Grad-CAM
![image](https://github.com/user-attachments/assets/7fe1f558-c975-4237-a83d-cbe70aac4039)

### Dice recognition
![image](https://github.com/user-attachments/assets/742dcc14-e02b-4c24-8ebb-db5cb78860f5)
![image](https://github.com/user-attachments/assets/f8da276d-4dbc-49ef-9d84-b5f5b159d567)

![image](https://github.com/user-attachments/assets/60775c2b-258d-4c81-8dfc-8707da783d22)
![image](https://github.com/user-attachments/assets/ca320031-2bb4-4a47-9eb5-cedee71031cc)


### YOLO (Dice localization and image crop) 

<img width="1066" alt="image-before-yolo" src="https://github.com/user-attachments/assets/e82d10b0-e611-44bd-a835-8038888d1935" />

![image](https://github.com/user-attachments/assets/eec36e9f-1fb1-4539-9d09-b64517311699)


