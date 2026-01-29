import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

st.title("Détecteur d'Or 🪙")

@st.cache_resource
def load_model():
    # On charge le modèle une seule fois pour ne pas figer l'écran
    return YOLO("yolov8n.pt")

model = load_model()

# Ajout d'une option pour uploader si la caméra bugge
option = st.radio("Source de l'image :", ("Appareil Photo", "Charger un fichier"))

if option == "Appareil Photo":
    img_file = st.camera_input("Prendre une photo")
else:
    img_file = st.file_uploader("Choisir une image", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    # Lecture de l'image
    image = Image.open(img_file)
    st.image(image, caption="Image capturée", use_container_width=True)
    
    with st.spinner("Analyse en cours..."):
        results = model(image)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Résultat de l'IA")
