import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import os
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect_maturation import detect_artificial_ripening

MODEL_ZIP_PATH = "../fruit_maturity_detector.zip"
MODEL_EXTRACT_PATH = "../fruit_maturity_detector.keras"
IMG_SIZE = 128

fruit_classes = ["ananas", "banane", "tomate", "papaye", "non_fruit"]
maturity_classes = ["pas_mur", "mur", "trop_mur"]

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

def extract_model(zip_path, extract_path):
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(extract_path))
    return extract_path

@st.cache_resource
def load_my_model(zip_path):
    model_path = extract_model(zip_path, MODEL_EXTRACT_PATH)
    return load_model(model_path)

model = load_my_model(MODEL_ZIP_PATH)

def predict_image(img_array):
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    pred_fruit, pred_maturity = model.predict(img)
    fruit_idx = int(np.argmax(pred_fruit))
    maturity_idx = int(np.argmax(pred_maturity))

    return fruit_idx, maturity_idx

st.set_page_config(page_title="Détecteur de Fruits", layout="centered")
st.title("🍌🍅🥭 Détecteur de Fruits & Maturité")
st.markdown("""
Téléchargez une image ou prenez une photo.  
Le modèle détecte si l'objet est un fruit (ananas, banane, tomate, papaye) ou non.  
S'il s'agit d'un fruit, il analyse aussi sa maturité.
""")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
use_camera = st.checkbox("📸 Utiliser la caméra")
img_array = None

if use_camera:
    picture = st.camera_input("Prenez une photo")
    if picture:
        file_bytes = np.frombuffer(picture.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        st.image(img_array, caption="Image capturée", use_container_width=True)

elif uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    st.image(img_array, caption="Image chargée", use_container_width=True)

if img_array is not None:
    fruit_idx, maturity_idx = predict_image(img_array)

    if fruit_classes[fruit_idx] == "non_fruit":
        text = "Ce n'est pas un fruit."
        st.warning(text)
        speak(text)
    else:
        text = f"C'est une **{fruit_classes[fruit_idx]}**, et elle est **{maturity_classes[maturity_idx]}**."
        st.success(text)
        speak(text)

        try:
            ripening_status = detect_artificial_ripening(img_array)
            st.subheader("🧪 Analyse supplémentaire : Mûrissement artificiel")
            st.write(f"Résultat : **{ripening_status}**")
        except Exception as e:
            st.error(f"Erreur lors de la détection du mûrissement artificiel : {e}")

st.markdown("---")
st.info("💡 Astuce : utilisez des images claires et centrées pour de meilleurs résultats.")
