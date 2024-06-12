import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from DigitModel import Digit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Digit().to(DEVICE)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 图片预处理
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img_tensor = torch.tensor(img, dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
    return img_tensor

def predict(model, image_tensor):
    output = model(image_tensor.to(DEVICE))
    _, predicted = torch.max(output, 1)
    return predicted.item()

image_path = 'OIP-C.jpeg'
image_tensor = preprocess_image(image_path)
predicted_digit = predict(model, image_tensor)
print(f'预测的数字是: {predicted_digit}')
