import cv2
import torch
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from DigitModel import Digit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Digit().to(DEVICE)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 图片预处理
def preprocess_image(img):
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img_tensor = torch.tensor(img, dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
    return img_tensor

#测试
def predict(model, image_tensor):
    output = model(image_tensor.to(DEVICE))
    _, predicted = torch.max(output, 1)
    return predicted.item()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    preprocessed_frame = preprocess_image(gray_frame)

    prediction = predict(model, preprocessed_frame)

    cv2.putText(frame, f'Predicted Digit: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Digit Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
