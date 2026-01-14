# Classificação de Qualidade de Carnes Utilizando Deep Learning e Transfer Learning

## Resumo

Este trabalho apresenta o desenvolvimento de um sistema de classificação automática da qualidade de carnes a partir de imagens digitais, utilizando técnicas de Aprendizado Profundo (Deep Learning). O objetivo é distinguir amostras de carne fresca e estragada por meio de Redes Neurais Convolucionais (CNNs), empregando Transfer Learning com a arquitetura MobileNetV2.

O pipeline proposto contempla desde o pré-processamento dos dados, divisão estratificada do dataset e treinamento em duas fases (warm-up e fine-tuning), até a avaliação quantitativa do modelo por meio de métricas clássicas de classificação. Os resultados demonstram que a abordagem adotada é eficaz, alcançando alta acurácia e boa capacidade de generalização.

## 1. Introdução

A avaliação da qualidade de carnes é uma etapa crítica na cadeia produtiva de alimentos, impactando diretamente a segurança alimentar e a saúde do consumidor. Métodos tradicionais de inspeção dependem fortemente da análise humana, estando sujeitos a subjetividade, fadiga e inconsistências.

Com o avanço do Aprendizado de Máquina e, em especial, do Aprendizado Profundo, tornou-se viável automatizar tarefas de classificação visual com alto grau de precisão. Neste contexto, este projeto propõe o uso de Redes Neurais Convolucionais para classificar imagens de carnes em dois estados: fresca (Fresh) e estragada (Spoiled), utilizando imagens como única fonte de informação.

## 2. Objetivos

### Objetivo Geral
Desenvolver um modelo baseado em Deep Learning capaz de classificar automaticamente a qualidade de carnes a partir de imagens digitais.

### Objetivos Específicos
- Implementar um pipeline de pré-processamento e carregamento de imagens.
- Aplicar a técnica de Transfer Learning utilizando a arquitetura MobileNetV2.
- Corrigir vieses de distribuição por meio de divisão estratificada do dataset.
- Avaliar o desempenho do modelo utilizando métricas quantitativas.

## 3. Metodologia

### 3.1 Dataset
O dataset é composto por imagens organizadas em duas classes:
- Fresh (carne fresca)
- Spoiled (carne estragada)

Devido ao tamanho do conjunto de dados, as imagens não estão incluídas neste repositório. A estrutura esperada é:
meat_dataset/
├── Fresh/
│ ├── image_01.jpg
│ └── ...
└── Spoiled/
├── image_01.jpg
└── ...

### 3.2 Pré-processamento e Split Estratificado
As imagens são redimensionadas e normalizadas conforme os requisitos da MobileNetV2. Para evitar viés estatístico, foi utilizada uma divisão estratificada dos dados, garantindo que 20% das amostras de cada classe fossem destinadas ao conjunto de validação.

### 3.3 Arquitetura do Modelo
Foi adotada a MobileNetV2 pré-treinada no ImageNet como backbone. Sobre ela, foram adicionadas camadas de:
- Global Average Pooling
- Camada densa com ativação ReLU
- Dropout para regularização
- Camada de saída com Softmax

## 4. Treinamento

O treinamento foi realizado em duas etapas:

### 4.1 Warm-up
Inicialmente, apenas as camadas densas adicionadas ao modelo foram treinadas, mantendo o backbone congelado. Essa etapa permite que o classificador se adapte ao novo domínio sem distorcer pesos pré-treinados.

### 4.2 Fine-tuning
Posteriormente, as camadas superiores da MobileNetV2 foram descongeladas, permitindo o ajuste fino das características aprendidas. O processo foi monitorado por meio da função de perda de validação, utilizando Early Stopping para evitar overfitting.

## 5. Avaliação e Resultados

O modelo final foi avaliado utilizando métricas clássicas de classificação, incluindo acurácia, precisão, recall e F1-score. A matriz de confusão foi gerada para análise detalhada dos erros.

### Resultados Obtidos
- Acurácia geral: **97.37%**
- Baixas taxas de falso-positivos e falso-negativos
- Boa capacidade de generalização para ambas as classes

Os artefatos gerados durante a avaliação estão disponíveis no diretório `results/`.

## 6. Estrutura do Repositório

meat-quality-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── training_experiment.ipynb
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── confusion_matrix.png
│   └── training_history.png
│
└── models/
    └── README.md

