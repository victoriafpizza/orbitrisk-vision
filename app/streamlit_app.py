import sys
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Permite importar arquivos da pasta src
sys.path.append(os.path.abspath("src"))

IMG_SIZE = (128, 128)
MODEL_PATH = "models/cnn_profunda.keras"

CLASS_NAMES = [
    "forest",
    "industrial",
    "residential",
    "river",
    "sealake"
]

CLASS_LABELS = {
    "forest": "Floresta / Vegetação",
    "industrial": "Área Industrial",
    "residential": "Área Residencial",
    "river": "Rio",
    "sealake": "Mar ou Lago"
}


@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model(MODEL_PATH)


def preparar_imagem(imagem):
    imagem = imagem.convert("RGB")
    imagem = imagem.resize(IMG_SIZE)
    imagem_array = np.array(imagem) / 255.0
    imagem_array = np.expand_dims(imagem_array, axis=0)
    return imagem_array


st.set_page_config(
    page_title="OrbitRisk Vision",
    page_icon="🛰️",
    layout="centered"
)

st.title("🛰️ OrbitRisk Vision")
st.subheader("Classificação de imagens satelitais com CNN")

st.write(
    "Envie uma imagem aérea ou satelital para o modelo classificar "
    "entre floresta, área industrial, área residencial, rio, mar ou lago."
)

modelo = carregar_modelo()

arquivo = st.file_uploader(
    "Escolha uma imagem para classificar:",
    type=["jpg", "jpeg", "png"]
)

if arquivo is not None:
    imagem = Image.open(arquivo)

    st.image(imagem, caption="Imagem enviada", use_container_width=True)

    imagem_processada = preparar_imagem(imagem)

    predicoes = modelo.predict(imagem_processada)[0]

    indice = np.argmax(predicoes)
    classe = CLASS_NAMES[indice]
    confianca = predicoes[indice] * 100

    st.success(f"Classe prevista: {CLASS_LABELS[classe]}")
    st.info(f"Confiança: {confianca:.2f}%")

    st.write("Probabilidade por classe:")

    probabilidades = {
        CLASS_LABELS[CLASS_NAMES[i]]: float(predicoes[i])
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(probabilidades)