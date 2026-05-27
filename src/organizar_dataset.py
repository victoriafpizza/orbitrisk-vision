import os
import shutil
import random


# As pastas originais estão na raiz do projeto:
# orbitrisk-vision/residential
# orbitrisk-vision/industrial
# orbitrisk-vision/forest
# orbitrisk-vision/river
# orbitrisk-vision/sealake
ORIGEM = "."

# Pasta de destino organizada para treino, validação e teste
DESTINO = "dataset"

# Classes que vamos usar
CLASSES = [
    "residential",
    "industrial",
    "forest",
    "river",
    "sealake"
]

# Quantidade exata de imagens por classe em cada divisão
QUANTIDADES = {
    "train": 200,
    "val": 40,
    "test": 40
}

# Para a divisão sempre sair igual
random.seed(42)


def limpar_pasta_destino():
    for divisao in QUANTIDADES.keys():
        for classe in CLASSES:
            pasta = os.path.join(DESTINO, divisao, classe)

            if os.path.exists(pasta):
                shutil.rmtree(pasta)

            os.makedirs(pasta, exist_ok=True)


def listar_imagens(pasta):
    extensoes_validas = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    imagens = [
        arquivo for arquivo in os.listdir(pasta)
        if arquivo.lower().endswith(extensoes_validas)
    ]

    return imagens


def organizar_dataset():
    limpar_pasta_destino()

    for classe in CLASSES:
        pasta_origem = os.path.join(ORIGEM, classe)

        if not os.path.exists(pasta_origem):
            print(f"ERRO: pasta não encontrada: {pasta_origem}")
            continue

        imagens = listar_imagens(pasta_origem)
        random.shuffle(imagens)

        total_necessario = sum(QUANTIDADES.values())

        if len(imagens) < total_necessario:
            print(f"ERRO: a classe '{classe}' tem apenas {len(imagens)} imagens.")
            print(f"São necessárias {total_necessario} imagens.")
            continue

        inicio = 0

        for divisao, quantidade in QUANTIDADES.items():
            selecionadas = imagens[inicio:inicio + quantidade]

            pasta_destino = os.path.join(DESTINO, divisao, classe)
            os.makedirs(pasta_destino, exist_ok=True)

            for imagem in selecionadas:
                caminho_origem = os.path.join(pasta_origem, imagem)
                caminho_destino = os.path.join(pasta_destino, imagem)

                shutil.copy2(caminho_origem, caminho_destino)

            print(f"{classe} -> {divisao}: {len(selecionadas)} imagens")

            inicio += quantidade

    print("\nDataset organizado com sucesso!")


organizar_dataset()