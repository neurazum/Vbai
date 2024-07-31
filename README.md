# Vbai Modelleri

## Model Detayları

#### Vbai Modelleri MRI ve fMRI görüntüleri üzerine eğitilmiştir. Bu modellerin eğitildiği veri setleri Neurazum tarafından gizli tutulmaktadır. Derin öğrenme yöntemleri ile eğitilerek çok yüksek doğruluk oranları ile MRI ve fMRI üzerinde çok hassas bir şekilde çalışabilir. Demans ile ilgili tüm beyin görselleriyle çalışıp, teşhis koyabilir. Nörobilim alanındaki geri kalmışlığa, ilkelliğe ve hata paylarına "bai" modelleriyle birlikte son vermeyi hedeflemektedir.

### Model Tanımı

- **Geliştirici:** _Neurazum_
- **Yayımcı:** _Eyüp İpler_
- **Model Tipi:** _MRI ve fMRI_
- **Lisans:** _CC-BY-NC-SA-4.0_

## Kullanımlar

**Bu modellerdeki amacımız;**

- _Hastanın demans hastalıklarını (alzheimer gibi) daha erken ve daha doğru bir şekilde teşhis koymak,_
- _Hastanelerde çalışan doktorlara teşhis ve inceleme için kolaylık sağlamak,_
- _Risk taşıyan hastaları tespit etmek,_
- _Tanı koyulma aşamasında ki hata paylarını düşürmektir._

## Direkt Kullanımlar

**Klasik Kullanım:**

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
model.load_state_dict(torch.load('Vbai-1.0 Dementia/model/yolu'))
model = model.to(device)
model.eval()
summary(model, (3, 224, 224))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Demans Değil', 'Hafif Demans', 'Orta Demans', 'Çok Hafif Demans']

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
    plt.title(f"Tahmin: {class_names[prediction]} (%{confidence * 100:.2f})")
    plt.axis('off')
    plt.show()

test_image_path = 'MRI/veya/fMRI/görüntüsü'
prediction, confidence = predict(test_image_path, model, transform)
print(f'Tahmin: {class_names[prediction]} (%{confidence * 100})')

show_image_with_prediction(test_image_path, prediction, confidence, class_names)
```

## Önyargı, Riskler ve Kısıtlamalar

**Vbai Modelleri;**

- _En büyük riski yanlış teşhis koymasıdır :),_
- _Herhangi bir kısıtlama bulunmamaktadır,_
- _Hastanın beyin görselleri hiçbir şekilde kişisel bilgi içermez. Bu nedenle, Vbai tarafından hiçbir şekilde kişisel veri elde edilemez._

### Öneriler

- _Görseller ne kadar yüksek çözünürlükte olursa o kadar iyidir._

## Modele Nasıl Başlanır

- Modelin içeriğindeki gerekli modülleri kurmak için;
- ```bash
    pip install -r requirements.txt
    ```
- Örnek kullanımla modelin ve veri setinin yolunu yerleştirin,
- Ve dosyayı çalıştırın.

## Değerlendirme

- Vbai-1.0 Dementia => (Doğruluk oranı en az her ihtimalde = %90) (DEMANS DURUMLARI)
- Vbai-1.1 Dementia => (Doğruluk oranı en az her ihtimalde = 90%) (DEMANS DURUMLARI)

### Sonuçlar

[![image](https://r.resimlink.com/BIgjLTN.png)](https://resimlink.com/BIgjLTN)

[![image](https://r.resimlink.com/3lQLUpatA.png)](https://resimlink.com/3lQLUpatA)

[![image](https://r.resimlink.com/uyT5Y.png)](https://resimlink.com/uyT5Y)

#### Özet

Özetle Vbai modelleri, hastanın demans durumunu tespit ederek tıp alanında çalışanlara kolaylık sağlamak amacıyla teşhis koyabilen görüntü işleme modelidir.

## Daha Fazla

LinkedIn: https://www.linkedin.com/company/neurazum

### Yazar

Eyüp İpler - https://www.linkedin.com/in/eyupipler/

### İletişim

neurazum@gmail.com

# ---------------------------------------

# Vbai Models

## Model Details

#### Vbai models were trained on MRI and fMRI images. The data sets on which these models are trained are kept confidential by Neurazum. It can work very precisely on MRI and fMRI with very high accuracy rates by training with deep learning methods. It can work with all brain images related to dementia and diagnose. It aims to put an end to the backwardness, primitiveness and error margins in the field of neuroscience with ‘bai’ models.

### Model Description

- **Developed by: _Neurazum_**
- **Shared by: _Eyüp İpler_**
- **Model type: _MRI and fMRI_**
- **License: _CC-BY-NC-SA-4.0_**

## Uses

**Our aim in these models is to;**

- _To diagnose the patient's dementia diseases (such as Alzheimer's) earlier and more accurately,_
- _Providing convenience to doctors working in hospitals for diagnosis and examination,_
- _Identifying patients at risk,_
- _to reduce the margin of error in the diagnostic process._

## Direct Uses

**Classical Use:**

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
model.load_state_dict(torch.load('Vbai-1.0 Dementia/model/path'))
model = model.to(device)
model.eval()
summary(model, (3, 224, 224))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Non Demented', 'Mild Demented', 'Moderate Demented', 'Very Mild Demented']

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

test_image_path = 'MRI/or/fMRI/image/path'
prediction, confidence = predict(test_image_path, model, transform)
print(f'Prediction: {class_names[prediction]} (%{confidence * 100})')

show_image_with_prediction(test_image_path, prediction, confidence, class_names)
```

## Bias, Risks and Limitations

**Vbai Models;**

- _The biggest risk is misdiagnosis :),_
- _There are no restrictions,_
- _The patient's brain images do not contain any personal information. Therefore, no personal data can be obtained by Vbai in any way._

### Recommendations

- _The higher the resolution of the visuals, the better._

## How to Get Started with the Model

- To install the necessary modeules in the model;
- ```bash 
    pip install -r requirements.txt
    ```
- Place the path of the model in the example uses.
- And run the file.

## Evaluation

- Vbai-1.0 Dementia => (Accuracy rate at least in all probability = 90%) (DEMENTIA STATES)
- Vbai-1.1 Dementia => (Accuracy rate at least in all probability = 90%) (DEMENTIA STATES)

### Results

[![image](https://r.resimlink.com/q93iSBueP0H.png)](https://resimlink.com/q93iSBueP0H)

[![image](https://r.resimlink.com/u5QMO0X42.png)](https://resimlink.com/u5QMO0X42)

[![image](https://r.resimlink.com/2NPDH0l.png)](https://resimlink.com/2NPDH0l)

#### Summary

In summary, Vbai models are image processing models that can diagnose the patient's dementia status in order to provide convenience to medical professionals.

## More

LinkedIn: https://www.linkedin.com/company/neurazum

### Author

Eyüp İpler - https://www.linkedin.com/in/eyupipler/

### Contact

neurazum@gmail.com
