# Dataset de Imagens de Carnes

Este diretório é reservado para o armazenamento do conjunto de dados utilizado no treinamento e avaliação dos modelos de aprendizado profundo deste projeto.

## Origem do Dataset

O conjunto de imagens utilizado neste trabalho foi obtido a partir do seguinte projeto público disponível na plataforma Kaggle:

**Meat Quality Assessment**  
Autor: Nalkrolu  
Link: https://www.kaggle.com/code/nalkrolu/meat-quality-assessment

O dataset contém imagens de carnes classificadas em dois estados de conservação:
- **Fresh**: carne em boas condições de consumo
- **Spoiled**: carne em estado de deterioração

## Licença e Uso dos Dados

As imagens pertencem aos respectivos autores originais conforme disponibilizado no Kaggle.  
Este repositório **não redistribui o dataset**, respeitando as diretrizes de uso e limitações de tamanho impostas pelo GitHub.

O uso das imagens neste projeto é estritamente acadêmico, sem fins comerciais, com o objetivo de estudo e pesquisa em Visão Computacional e Aprendizado de Máquina.

## Estrutura Esperada do Dataset

Após o download e extração do dataset, os dados devem ser organizados localmente da seguinte forma:
```
data/
└── meat_dataset/
├── Fresh/
│ ├── image_001.jpg
│ ├── image_002.jpg
│ └── ...
└── Spoiled/
├── image_001.jpg
├── image_002.jpg
└── ...
```

Essa estrutura é **obrigatória** para a correta execução dos scripts de treinamento e avaliação presentes neste repositório.

## Observações Importantes

- As imagens **não são versionadas** neste repositório devido ao seu tamanho.
- O caminho para o dataset pode ser configurado nos scripts de treinamento conforme o ambiente de execução (local ou Google Colab).
- Recomenda-se verificar a integridade das imagens antes do treinamento, garantindo que não existam arquivos corrompidos ou formatos incompatíveis.

## Reprodutibilidade

Qualquer pesquisador ou avaliador pode reproduzir os experimentos descritos neste projeto seguindo os passos:
1. Acessar o link do dataset no Kaggle.
2. Realizar o download das imagens.
3. Organizar o dataset conforme a estrutura descrita acima.
4. Executar os notebooks ou scripts presentes no diretório `src/`.

Essa abordagem garante transparência, reprodutibilidade científica e conformidade com boas práticas acadêmicas.
