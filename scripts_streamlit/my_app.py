import streamlit as st
import numpy as np
import cv2
import os
from keras.models import load_model
from ultralytics import YOLO
import sys

# YOLO pour la détection des fruits
yolo_model = YOLO("yolov8n.pt")  # modèle léger et rapide

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect_maturation import detect_artificial_ripening

MODEL_PATH = "../fruit_maturity_detector.keras"
IMG_SIZE = 128

fruit_classes = ["ananas", "banane", "tomate", "papaye", "non_fruit"]
maturity_classes = ["pas_mur", "mur", "trop_mur"]
fruit_genre = {
    "ananas": "un",
    "banane": "une",
    "tomate": "une",
    "papaye": "une"
}

# Mapping YOLO COCO → fruit pour notre modèle
yolo_to_fruit = {
    46: "banane",
    49: "tomate"
}

def speak(text):
    st.components.v1.html(
        f"""
        <script>
            var msg = new SpeechSynthesisUtterance("{text}");
            window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

def predict_image(img_array):
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    pred_fruit, pred_maturity = model.predict(img)
    fruit_idx = int(np.argmax(pred_fruit))
    maturity_idx = int(np.argmax(pred_maturity))

    return fruit_idx, maturity_idx

st.set_page_config(page_title="Détecteur de Fruits", layout="centered")
st.title("🍌🍅 Détecteur de fruit & Maturité")
st.markdown("""
Téléchargez une image ou prenez une photo pour détecter les fruits.  
YOLO analyse l'image d'abord, puis le modèle de maturité traite chaque fruit détecté.
""")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
use_camera = st.checkbox("📸 Utiliser la caméra")
img_path = None

if use_camera:
    picture = st.camera_input("Prenez une photo")
    if picture:
        img_path = "temp_img.jpg"
        with open(img_path, "wb") as f:
            f.write(picture.getbuffer())
elif uploaded_file:
    img_path = f"temp_upload_{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

if img_path:
    results = yolo_model(img_path)
    img_array = cv2.imread(img_path)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    st.image(img_array, caption="Image analysée", use_container_width=True)

    fruits_detected = False

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls in yolo_to_fruit:
            fruits_detected = True
            fruit_name = yolo_to_fruit[cls]
            genre = fruit_genre.get(fruit_name, "un")

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img_array[y1:y2, x1:x2]

            fruit_idx, maturity_idx = predict_image(crop)
            maturity = maturity_classes[maturity_idx]

            st.subheader(f"📍 Fruit détecté : {fruit_name}")
            st.image(crop, caption=f"Zone analysée ({fruit_name})", use_container_width=True)

            result_text = f"C'est {genre} {fruit_name}, et elle est {maturity}"
            st.success(result_text)
            speak(result_text)

            try:
                ripening_status = detect_artificial_ripening(crop)
                st.write(f"🧪 Mûrissement artificiel : {ripening_status}")
            except Exception as e:
                st.error(f"Erreur lors de l'analyse du mûrissement artificiel : {e}")

    if not fruits_detected:
        st.warning("Aucun fruit reconnu par YOLO.")
        speak("Aucun fruit détecté")

st.markdown("---")
st.info("💡 Astuce : utilisez des images claires et centrées pour de meilleurs résultats.")
