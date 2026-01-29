import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="GoldDetector Pro", layout="centered")

st.title("🪙 Détecteur d'Or Haute Qualité")

@st.cache_resource
def load_model():
    # Chargement du modèle
    return YOLO("yolov8n.pt")

model = load_model()

st.write("### 📸 Consigne pour la qualité maximale :")
st.info("Clique sur le bouton ci-dessous, puis choisis **'Appareil Photo'**. Active ton flash manuellement et utilise le mode Macro si disponible.")

# Utilisation de file_uploader au lieu de camera_input pour forcer l'app native
img_file = st.file_uploader("Prendre une photo (Flash & HD)", type=['jpg', 'png', 'jpeg'])

if img_file is not None:
    # Traitement de l'image haute résolution
    image = Image.open(img_file)
    
    # Affichage de l'image originale
    st.image(image, caption="Image source reçue", use_container_width=True)
    
    with st.spinner("Analyse de précision en cours..."):
        # Inférence
        results = model(image)
        
        # Affichage des résultats
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Résultat de l'analyse IA", use_container_width=True)
        
        if len(results[0].boxes) == 0:
            st.warning("Aucun objet détecté. Assurez-vous d'avoir un bon éclairage avec le flash.")
