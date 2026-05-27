# OrbitRisk Vision

Projeto desenvolvido para a Global Solution, na disciplina de Computer Vision.

## Integrante

- Victoria Franceschini Pizza — RM550609

## Descrição do Projeto

O OrbitRisk Vision é uma solução de Visão Computacional aplicada ao contexto da Indústria Espacial.

O objetivo do projeto é classificar imagens satelitais de uso do solo, identificando automaticamente diferentes tipos de áreas observadas por sensores orbitais.

A solução utiliza redes neurais convolucionais, treinadas do zero, para classificar imagens em cinco categorias:

- Forest — Floresta / Vegetação
- Industrial — Área Industrial
- Residential — Área Residencial
- River — Rio
- SeaLake — Mar ou Lago

Essa classificação pode apoiar sistemas de monitoramento territorial, análise ambiental, planejamento urbano e tomada de decisão baseada em imagens de satélite.

## Conexão com a Indústria Espacial

O projeto está conectado à Indústria Espacial por utilizar imagens satelitais como base para análise automática do território.

Satélites de observação da Terra são amplamente utilizados para monitoramento ambiental, urbano e geográfico. A solução proposta simula um módulo inteligente capaz de interpretar imagens captadas por sensores orbitais e classificar o tipo de área presente na imagem.

## Dataset

Foi utilizado o dataset EuroSAT, composto por imagens satelitais obtidas a partir do satélite Sentinel-2.

Para este projeto, foram selecionadas 5 classes:

- Forest
- Industrial
- Residential
- River
- SeaLake

A divisão dos dados foi feita da seguinte forma:

| Conjunto | Imagens por classe | Total |
|---|---:|---:|
| Treino | 200 | 1000 |
| Validação | 40 | 200 |
| Teste | 40 | 200 |

Total utilizado no projeto: 1400 imagens.

## Pré-processamento

As imagens foram redimensionadas para o tamanho de 128x128 pixels e normalizadas para valores entre 0 e 1.

Durante o treinamento, foram aplicadas técnicas de data augmentation no conjunto de treino:

- Rotação
- Zoom
- Deslocamento horizontal
- Deslocamento vertical
- Espelhamento horizontal

Essas técnicas ajudam o modelo a generalizar melhor e reduzem o risco de overfitting.

## Arquiteturas Utilizadas

Foram criadas e treinadas duas redes neurais convolucionais do zero, sem uso de modelos pré-treinados.

### CNN Simples

A primeira arquitetura foi criada como modelo base, contendo:

- Camadas Conv2D
- Camadas MaxPooling2D
- Camada Flatten
- Camada Dense
- Dropout
- Camada final Softmax

Essa arquitetura é mais simples, com menor profundidade e menor custo computacional.

### CNN Profunda

A segunda arquitetura possui maior profundidade e capacidade de aprendizado, contendo:

- Mais blocos convolucionais
- MaxPooling2D
- Camadas densas maiores
- Regularização L2
- Dropout
- Camada final Softmax

Essa arquitetura foi criada para aprender padrões visuais mais complexos nas imagens satelitais.

## Resultados

Os modelos foram avaliados no conjunto de teste, contendo 200 imagens.

| Modelo | Acurácia no teste | Loss no teste |
|---|---:|---:|
| CNN Simples | 89% | 0.2756 |
| CNN Profunda | 97% | 0.2226 |

A CNN Profunda apresentou melhor desempenho, alcançando 97% de acurácia no conjunto de teste.

## Comparação dos Modelos

A CNN Simples apresentou um bom resultado, superando a referência mínima de 88% de acurácia.

Porém, a CNN Profunda obteve desempenho superior por possuir mais camadas convolucionais e maior capacidade de extração de padrões visuais.

A arquitetura mais profunda conseguiu identificar melhor diferenças entre áreas industriais, residenciais, vegetação, rios e lagos, resultando em maior precisão geral.

## Métricas Geradas

Durante o projeto foram gerados:

- Gráficos de acurácia
- Gráficos de loss
- Matriz de confusão
- Relatório de classificação
- Teste com imagens novas
- Aplicação funcional com Streamlit

Os arquivos de resultado estão na pasta:

results/

## Demonstração Funcional

A demonstração foi desenvolvida com Streamlit.

A aplicação permite que o usuário envie uma imagem e receba como saída:

- Imagem enviada
- Classe prevista
- Porcentagem de confiança
- Gráfico de probabilidade por classe

Para executar a aplicação:

streamlit run app/streamlit_app.py

## Como Executar o Projeto

### 1. Criar ambiente virtual

python -m venv venv

### 2. Ativar ambiente virtual

No Windows:

venv\Scripts\activate.bat

### 3. Instalar dependências

pip install -r requirements.txt

### 4. Organizar dataset

python src/organizar_dataset.py

### 5. Treinar os modelos

python src/train.py

### 6. Avaliar os modelos

python src/evaluate.py

### 7. Testar predição

python src/predict.py

### 8. Rodar aplicação Streamlit

streamlit run app/streamlit_app.py

## Tecnologias Utilizadas

- Python
- TensorFlow / Keras
- Scikit-learn
- Matplotlib
- NumPy
- Pillow
- Streamlit
- VS Code
- GitHub

## Conclusão

O projeto demonstrou a aplicação prática de Visão Computacional no contexto da Indústria Espacial, utilizando imagens satelitais para classificação de uso do solo.

Foram treinadas duas CNNs do zero, comparadas por métricas quantitativas e avaliadas com matriz de confusão.

A melhor arquitetura alcançou 97% de acurácia, superando a meta mínima de 88%.

A aplicação em Streamlit permite demonstrar o modelo funcionando com novas imagens, tornando a solução prática, visual e adequada para apresentação da Global Solution.

## Link do Vídeo

Adicionar aqui o link do vídeo no YouTube:

COLOCAR_LINK_DO_VIDEO_AQUI