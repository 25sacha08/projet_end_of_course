import streamlit as st
import numpy as np
import cv2
from keras.models import load_model


MODEL_PATH = "../fruit_maturity_detector.keras"
IMG_SIZE = 128
fruit_classes = ["banane", "tomate", "non_fruit"]
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

@st.cache_resource
def load_my_model(path):
    return load_model(path)

model = load_my_model(MODEL_PATH)

def predict_image(img_array):
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred_fruit, pred_maturity = model.predict(img)
    fruit_idx = np.argmax(pred_fruit)
    maturity_idx = np.argmax(pred_maturity)
    return fruit_idx, maturity_idx

st.set_page_config(page_title="Détecteur de fruits", layout="centered")

st.title("Détecteur de Fruits et Maturité")
st.markdown("""
**Instructions :**
- Chargez une image ou utilisez votre caméra.
- Le modèle détectera si c'est un fruit (`banane` ou `tomate`) ou non.
- Si c'est un fruit, il donnera également son niveau de maturité.
""")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

use_camera = st.checkbox("Utiliser la caméra")

if use_camera:
    picture = st.camera_input("Prenez une photo")
    if picture:
        file_bytes = np.frombuffer(picture.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        st.image(img_array, caption="Image capturée", use_container_width=True)

        fruit_idx, maturity_idx = predict_image(img_array)

        if fruit_classes[fruit_idx] == "non_fruit":
            text = "Ce n'est pas un fruit."
            st.warning("error" + text)
        else:
            text = f"C'est une {fruit_classes[fruit_idx]}, elle est {maturity_classes[maturity_idx]}."
            st.success("ok" + text)

        speak(text)

elif uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    st.image(img_array, caption="Image chargée", use_container_width=True)

    fruit_idx, maturity_idx = predict_image(img_array)

    if fruit_classes[fruit_idx] == "non_fruit":
        text = "Ce n'est pas un fruit."
        st.warning("error" + text)
    else:
        text = f"C'est une {fruit_classes[fruit_idx]}, elle est {maturity_classes[maturity_idx]}."
        st.success("ok" + text)

    speak(text)

st.markdown("---")
st.info("💡 Astuce : pour de meilleurs résultats, utilisez des images claires avec le fruit bien visible.")
