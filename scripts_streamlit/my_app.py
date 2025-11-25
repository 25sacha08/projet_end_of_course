# import streamlit as st

# st.write("Hello world")

# # Afficher un widget pour capturer une image
# picture = st.camera_input("Prenez une photo")

# # Si une image est capturée, l'afficher
# if picture:
#     st.image(picture)   

import streamlit as st
import numpy as np
from keras.models import load_model
import cv2

# if st.button("Activer la caméra"):
#     st.session_state.camera_enabled = True
# elif st.button("Désactiver la caméra"):
#     st.session_state.camera_enabled = False

# if 'camera_enabled' not in st.session_state:
#     st.session_state.camera_enabled = False

# picture = st.camera_input("Prendre une photo", disabled=not st.session_state.camera_enabled)


# uploaded_file = st.file_uploader("Choose a image file", type="jpg")
import streamlit as st
from keras.models import load_model
import numpy as np
import cv2

model = load_model("../fruit_model.h5")
CLASSES = ["pas_mur", "mur", "trop_mur"]

st.title("Test de classification de tomate")

uploaded = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(show, caption="Image chargée")

    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred)

    st.success(f"Résultat : {CLASSES[class_idx]}")

# if picture:
#     st.image(picture)   
# model = load_model("../banana_model.h5")
# img = cv2.imread(picture)
# classes = ["pas_mur", "mur", "trop_mur"]


#     # continue
# if img is None:
#     print(f"Failed to load image: {img}")
# else:
#     resized_img = cv2.resize(img, (64, 64))   
# # img = cv2.resize(img, (128, 128))
#     img = resized_img / 255.0
#     img = np.expand_dims(img, axis=0)

#     pred = model.predict(img)
#     class_idx = np.argmax(pred)

#     print("Résultat :", classes[class_idx])