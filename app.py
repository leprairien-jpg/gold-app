import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Gold Detector", layout="centered")

st.title("🪙 Détecteur d'Or Prototype")

# Chargement du modèle YOLOv8 nano (le plus léger pour mobile)
# Il sera téléchargé automatiquement au premier lancement
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt") 

model = load_model()

# Interface de capture
img_file = st.camera_input("Prendre une photo d'un objet")

if img_file:
    img = Image.open(img_file)
    
    # Inférence (L'IA analyse)
    results = model(img)
    
    # Affichage
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Analyse en cours...", use_container_width=True)
    
    # Message d'aide
    st.warning("Note : Ce modèle est générique. Pour détecter spécifiquement l'or pur, vous devrez fournir vos propres photos pour l'entraînement final.")
