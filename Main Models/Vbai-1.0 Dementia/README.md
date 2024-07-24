# Vbai-1.0 Dementia (11178564 parametre)

## "Vbai-1.0 Dementia" modeli, hastanın demans durumunu MRI ve fMRI görüntüleri üzerine teşhis edebilecek şekilde eğitildi.

## -----------------------------------------------------------------------------------

# Vbai-1.0 Dementia (11178564 parameters)

## The "Vbai-1.0 Dementia" model has been trained to diagnose the patient's dementia condition on MRI and fMRI images.

# Kullanım / Usage

```python
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model.load_state_dict(torch.load('Vbai-1.0 Dementia/path'))
model = model.to(device)
model.eval()
summary(model, (3, 224, 224))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['No Dementia', 'Mild Dementia', 'Avarage Dementia', 'Very Mild Dementia']

def predict(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    return preds.item(), probs[0][preds.item()].item()

def show_image_with_prediction(image_path, prediction, confidence, class_names):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f"Prediction: {class_names[prediction]} (%{confidence * 100:.2f})")
    plt.axis('off')
    plt.show()

test_image_path = 'image-path'
prediction, confidence = predict(test_image_path, model, transform)
print(f'Prediction: {class_names[prediction]} (%{confidence * 100})')

show_image_with_prediction(test_image_path, prediction, confidence, class_names)
```

# Uygulama / As App

```python
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['No Dementia', 'Mild Dementia', 'Avarage Dementia', 'Very Mild Dementia']


class DementiaApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = None
        self.image_path = None

    def initUI(self):
        self.setWindowTitle('Prediction App by Neurazum')
        self.setWindowIcon(QIcon('C:/Users/eyupi/PycharmProjects/Neurazum/NeurAI/Assets/neurazumicon.ico'))
        self.setGeometry(2500, 300, 400, 200)

        self.loadModelButton = QPushButton('Upload Model', self)
        self.loadModelButton.clicked.connect(self.loadModel)

        self.loadImageButton = QPushButton('Upload Image', self)
        self.loadImageButton.clicked.connect(self.loadImage)

        self.predictButton = QPushButton('Make a Prediction', self)
        self.predictButton.clicked.connect(self.predict)
        self.predictButton.setEnabled(False)

        self.resultLabel = QLabel('', self)
        self.resultLabel.setAlignment(Qt.AlignCenter)

        self.imageLabel = QLabel('', self)
        self.imageLabel.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.loadModelButton)
        layout.addWidget(self.loadImageButton)
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.predictButton)
        layout.addWidget(self.resultLabel)

        self.setLayout(layout)

    def loadModel(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose Model Path", "",
                                                  "PyTorch Model Files (*.pt);;All Files (*)", options=options)
        if fileName:
            self.model = models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 4)
            self.model.load_state_dict(torch.load(fileName, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
            self.predictButton.setEnabled(True)
            QMessageBox.information(self, "Model Uploaded", "Model successfully uploaded!")

    def loadImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose Image File", "",
                                                  "Image Files (*.jpg *.jpeg *.png);;All Files (*)", options=options)
        if fileName:
            self.image_path = fileName
            pixmap = QPixmap(self.image_path)
            self.imageLabel.setPixmap(pixmap.scaled(224, 224, Qt.KeepAspectRatio))

    def predict(self):
        if self.model and self.image_path:
            prediction, confidence = self.predictImage(self.image_path, self.model, transform)
            self.resultLabel.setText(f'Prediction: {class_names[prediction]} (%{confidence * 100:.2f})')
        else:
            QMessageBox.warning(self, "Missing Information", "Model and picture must be uploaded.")

    def predictImage(self, image_path, model, transform):
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
        return preds.item(), probs[0][preds.item()].item()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DementiaApp()
    ex.show()
    sys.exit(app.exec_())
```

# Python Sürümü / Python Version

### 3.9 <=> 3.13

# Modüller / Modules

```bash
matplotlib==3.8.0
Pillow==10.0.1
torch==2.3.1
torchsummary==1.5.1
torchvision==0.18.1
```