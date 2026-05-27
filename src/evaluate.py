import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


IMG_SIZE = (128, 128)
BATCH_SIZE = 32

TEST_DIR = "dataset/test"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())


def avaliar_modelo(nome_modelo):
    caminho_modelo = os.path.join(MODELS_DIR, f"{nome_modelo}.keras")

    print(f"\nAvaliando modelo: {nome_modelo}")

    modelo = tf.keras.models.load_model(caminho_modelo)

    loss, accuracy = modelo.evaluate(test_generator)

    print(f"Loss no teste: {loss:.4f}")
    print(f"Acurácia no teste: {accuracy:.4f}")

    probabilidades = modelo.predict(test_generator)
    y_pred = np.argmax(probabilidades, axis=1)
    y_true = test_generator.classes

    print("\nRelatório de classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    matriz = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=class_names
    )

    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    caminho_matriz = os.path.join(
        RESULTS_DIR,
        f"matriz_confusao_{nome_modelo}.png"
    )

    plt.savefig(caminho_matriz)
    plt.close()

    print(f"Matriz de confusão salva em: {caminho_matriz}")


avaliar_modelo("cnn_simples")
avaliar_modelo("cnn_profunda")

print("\nAvaliação finalizada com sucesso!")