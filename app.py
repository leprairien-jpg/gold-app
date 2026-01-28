import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Reconnaissance d'Or 🪙")

# On charge un modèle de test (YOLOv8 nano)
model = YOLO("yolov8n.pt") 

img_file = st.camera_input("Scanner un objet")

if img_file:
    img = Image.open(img_file)
    results = model(img) # L'IA analyse l'image

    # Affichage du résultat
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Analyse terminée")
