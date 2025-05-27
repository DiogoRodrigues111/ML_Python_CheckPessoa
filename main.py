import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity


# Carregar modelo pré-treinado (ResNet18 como extrator de features)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # Remover camada de classificação
model.eval()

# Pré-processamento
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean do ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

# Carregar Haar Cascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Banco de embeddings (Exemplo com uma pessoa cadastrada)
database = {}

def get_embedding(face_img):
    img_tensor = preprocess(face_img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor).numpy()
    return embedding

# Cadastro de rosto
def register_face(label, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        embedding = get_embedding(face)
        database[label] = embedding
        print(f"Registrado {label}")
        return

# Comparar rosto
def recognize_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        embedding = get_embedding(face)

        label = "Desconhecido"
        max_sim = 0

        for name, db_embed in database.items():
            sim = cosine_similarity(embedding, db_embed)[0][0]
            if sim > max_sim and sim > 0.6:  # Threshold
                max_sim = sim
                label = name

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return img

# ----------------
# Teste Webcam
# ----------------
cap = cv2.VideoCapture(0)

# Cadastrar uma pessoa (capture o primeiro frame)
ret, frame = cap.read()
register_face("Pessoa1", frame)

while True:
    ret, frame = cap.read()
    frame = recognize_face(frame)
    cv2.imshow('Reconhecimento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
