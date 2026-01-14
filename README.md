# Classificação de Qualidade de Carnes Utilizando Deep Learning e Transfer Learning

## Resumo

Este projeto apresenta uma abordagem baseada em Aprendizado de Máquina e Visão Computacional para a classificação automática da qualidade de carnes a partir de imagens digitais. O problema abordado consiste em distinguir amostras de carne fresca (Fresh) e estragada (Spoiled), utilizando Redes Neurais Convolucionais (CNNs) com a técnica de Transfer Learning.

Foi empregada a arquitetura MobileNetV2, previamente treinada no conjunto ImageNet, combinada com um classificador ajustado ao domínio específico do problema. O estudo contempla desde o pré-processamento dos dados até o ajuste fino do modelo (fine-tuning), com atenção especial à correção de vieses estatísticos por meio de divisão estratificada dos dados. Os resultados experimentais demonstram alta capacidade de generalização, alcançando 97,37% de acurácia no conjunto de validação.

## 1. Introdução

A avaliação da qualidade de carnes é um fator crítico na indústria alimentícia, impactando diretamente a segurança do consumidor, a redução de desperdícios e a padronização de processos. Métodos tradicionais baseiam-se majoritariamente em inspeção humana, sujeita a subjetividade, fadiga e variação entre avaliadores.

Com o avanço das técnicas de Aprendizado Profundo, especialmente no campo da visão computacional, tornou-se viável a aplicação de modelos capazes de extrair automaticamente características visuais relevantes a partir de imagens. Nesse contexto, este trabalho propõe o uso de uma CNN com Transfer Learning para automatizar a classificação da qualidade da carne, explorando representações profundas já consolidadas em grandes bases de dados.

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
```
meat_dataset/
├── Fresh/
│   ├── image_01.jpg
│   ├── image_02.jpg
│   └── ...
└── Spoiled/
    ├── image_01.jpg
    ├── image_02.jpg
    └── ...
```
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
```
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
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── classification_report.txt
│   └── evaluation_metrics.txt
│
└── models/
    └── README.md
```
## 7. Como Reproduzir os Resultados

### 1. Clonar o repositório:
git clone https://github.com/seu-usuario/meat-quality-classification.git
cd meat-quality-classification

### 2. Criar ambiente virtual e instalar dependências:
pip install -r requirements.txt

### 3. Organizar o dataset conforme descrito.

### 4. Executar o treinamento:
python src/train.py

### 5. Avaliar o modelo:
python src/evaluate.py


## 8. Limitações e Trabalhos Futuros

Apesar dos resultados promissores, o projeto apresenta algumas limitações:
- Classificação restrita a duas classes.
- Dependência de imagens com boa iluminação e enquadramento.
- Dataset relativamente limitado.

Como trabalhos futuros, sugere-se:
- Expansão para múltiplos níveis de qualidade.
- Teste com outras arquiteturas (EfficientNet, ResNet).
- Aplicação em ambiente industrial em tempo real.
- Integração com sistemas embarcados ou edge computing.

## 9. Conclusão

Este trabalho demonstrou que técnicas modernas de Deep Learning, aliadas ao Transfer Learning, são capazes de fornecer soluções eficazes para problemas reais da indústria alimentícia. A organização modular do código, aliada à metodologia experimental rigorosa, torna o projeto reprodutível, extensível e adequado a contextos acadêmicos e profissionais.

## Licença

Este projeto é distribuído sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
