import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from models import criar_cnn_simples, criar_cnn_profunda


IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

RESULTS_DIR = "results"
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def criar_geradores():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator


def plotar_historico(history, nome_modelo):
    plt.figure()
    plt.plot(history.history["accuracy"], label="Treino")
    plt.plot(history.history["val_accuracy"], label="Validação")
    plt.title(f"Acurácia - {nome_modelo}")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"accuracy_{nome_modelo}.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title(f"Loss - {nome_modelo}")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"loss_{nome_modelo}.png"))
    plt.close()


def treinar_modelo(modelo, nome_modelo, train_generator, val_generator):
    modelo.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{nome_modelo}.keras"),
            monitor="val_accuracy",
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )
    ]

    history = modelo.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    plotar_historico(history, nome_modelo)

    return history


train_generator, val_generator = criar_geradores()

cnn_simples = criar_cnn_simples()
history_simples = treinar_modelo(
    cnn_simples,
    "cnn_simples",
    train_generator,
    val_generator
)

cnn_profunda = criar_cnn_profunda()
history_profunda = treinar_modelo(
    cnn_profunda,
    "cnn_profunda",
    train_generator,
    val_generator
)

print("Treinamento finalizado com sucesso!")