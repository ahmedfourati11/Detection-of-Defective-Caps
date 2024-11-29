import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model import Modele_detection_of_defective_caps  

# Charger le modèle
model = Modele_detection_of_defective_caps()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Transformation des images pour correspondre aux attentes du modèle
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Détection des cercles
def detect_circle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)  # Dessine un cercle autour du bouchon
            return frame, (x, y, r)
    return frame, None

# Fonction pour la prédiction
def predict(frame):
    image = Image.fromarray(frame)  
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        output = model(image)
        predicted_prob = torch.sigmoid(output).item()
        predicted_class = int(predicted_prob > 0.5)
        return predicted_class, predicted_prob

# Accéder à la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Impossible d'accéder à la caméra")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de capture d'image")
        break

    # Détecter les cercles (bouchons potentiels)
    frame, circle = detect_circle(frame)
    if circle is not None:
        x, y, r = circle
        # Recadrer autour du cercle détecté
        cropped = frame[y-r:y+r, x-r:x+r]
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:  
            pred_class, prob = predict(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            label = f"Défectueux: {prob:.2f}" if pred_class == 1 else f"Intact: {1-prob:.2f}"
            # Afficher le label près du bouchon détecté
            cv2.putText(frame, label, (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    cv2.imshow("Détection de bouchon", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
