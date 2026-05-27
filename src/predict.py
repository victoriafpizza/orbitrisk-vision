import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


IMG_SIZE = (128, 128)
MODEL_PATH = "models/cnn_profunda.keras"

CLASS_NAMES = [
    "forest",
    "industrial",
    "residential",
    "river",
    "sealake"
]


modelo = tf.keras.models.load_model(MODEL_PATH)


def prever_imagem(caminho_imagem):
    img = image.load_img(caminho_imagem, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predicoes = modelo.predict(img_array)

    indice_classe = np.argmax(predicoes[0])
    classe_prevista = CLASS_NAMES[indice_classe]
    confianca = predicoes[0][indice_classe] * 100

    return classe_prevista, confianca, predicoes[0]


if __name__ == "__main__":
    caminho_teste = "dataset/test/forest"

    primeira_imagem = os.listdir(caminho_teste)[0]
    caminho_completo = os.path.join(caminho_teste, primeira_imagem)

    classe, confianca, probabilidades = prever_imagem(caminho_completo)

    print(f"Imagem testada: {caminho_completo}")
    print(f"Classe prevista: {classe}")
    print(f"Confiança: {confianca:.2f}%")
    print("Probabilidades:", probabilidades)