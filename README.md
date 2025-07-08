# Classificador de Filmes IA
Integrantes: 
Yálamo Thiago, 
Robert Danilo,
Alicia Monteiro,
Kleiton Josivan,
Ciro Assuero,
João Vitor Fernandes.

Este projeto é um classificador probabilístico de filmes baseado em notas do IMDb. Ele usa a descrição e o gênero de um filme para prever a probabilidade de ele ter uma nota alta (>= 7.0) ou baixa (< 7.0).

## Como Usar

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/yalamothiago/classificador-filmes](https://github.com/yalamothiago/classificador-filmes)
    ```
   
2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Baixe os dados do NLTK:**
    ```bash
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
    ```


4.  **Execute o script:**
    ```bash
    python recomendador_probabilistico.py
    ```
  

## Definição de "Bom" Filme

Neste projeto, um filme é considerado "bom" se sua nota no IMDb for maior ou igual a 7.0.

## Créditos

* Dataset: Kaggle (IMDB)
