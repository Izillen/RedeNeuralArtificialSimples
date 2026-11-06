# 🧠 Rede Neural Simples com Python

Implementação de um modelo de aprendizado de máquina capaz de aprender uma função linear a partir de dados, utilizando **Python, NumPy e Gradient Descent**.

## 📌 Sobre o projeto

Este projeto tem como objetivo explorar os fundamentos do treinamento de modelos de Machine Learning, implementando manualmente os principais componentes de um modelo de regressão linear.

O modelo recebe dados de entrada, realiza previsões, calcula o erro e ajusta seus parâmetros iterativamente até encontrar uma reta que represente adequadamente os dados fornecidos.

## ⚙️ Tecnologias utilizadas

- **Python** – Linguagem de programação.
- **NumPy** – Manipulação de arrays e operações matemáticas.
- **Matplotlib** – Visualização gráfica dos dados e resultados.

## 🧩 Conceitos explorados

- Regressão Linear
- Forward Propagation
- Mean Squared Error (MSE)
- Gradient Descent
- Cálculo de gradientes
- Ajuste de pesos (`w`) e bias (`b`)
- Treinamento por épocas
- Visualização de resultados

## 🚀 Funcionamento

O modelo aprende uma função linear definida por:

\[
\hat{Y} = wx + b
\]

Onde:

- `x`: dados de entrada.
- `w`: peso do modelo.
- `b`: bias.
- `ŷ`: valor previsto.

Durante o treinamento, o modelo utiliza o erro quadrático médio (MSE) para avaliar suas previsões e Gradient Descent para ajustar os parâmetros.

O processo é repetido durante várias épocas, buscando minimizar o erro e aproximar os parâmetros aprendidos dos valores reais.

## 📊 Visualização

O projeto gera dados sintéticos seguindo uma função linear com ruído aleatório e apresenta um gráfico comparando:

- Os dados reais utilizados no treinamento.
- A reta aprendida pelo modelo.

## 🛠️ Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
```

### 2. Acesse a pasta do projeto

```bash
cd SEU-REPOSITORIO
```

### 3. Instale as dependências

```bash
pip install numpy matplotlib
```

### 4. Execute o projeto

```bash
python main.py
```

## 🎯 Objetivo de aprendizado

Projeto desenvolvido com finalidade acadêmica e de estudo, buscando compreender na prática como ocorre o aprendizado de um modelo de Machine Learning, desde a inicialização dos parâmetros até a otimização por Gradient Descent.


