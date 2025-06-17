import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # Para lematização
from sklearn.naive_bayes import MultinomialNB # Para Naive Bayes

# Certifique-se de baixar as stopwords e o wordnet do NLTK uma vez:
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4') # Open Multilingual Wordnet, útil para lematização

# --- Funções Auxiliares ---
def limpar_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Inicializa o lematizador fora da função para melhor performance
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Limpeza de texto e lematização:
    - Converte para minúsculas.
    - Remove caracteres não alfanuméricos (mantém espaços).
    - Remove números.
    - Remove stopwords em inglês E português.
    - Aplica lematização.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove caracteres não alfabéticos
    text = re.sub(r'\d+', '', text) # Remove números

    words = text.split()
    
    # Carrega stopwords de inglês e português
    english_stopwords = set(stopwords.words('english'))
    portuguese_stopwords = set(stopwords.words('portuguese'))
    all_stopwords = english_stopwords.union(portuguese_stopwords)

    # Remove stopwords e aplica lematização
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in all_stopwords]
    text = ' '.join(filtered_words)

    return text

# --- 1. Carregamento e Pré-processamento de Dados do Kaggle ---
print("--- Carregando e Pré-processando Dados do Dataset IMBD ---")

file_path = 'IMBD.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Dataset carregado com sucesso de '{file_path}'")
    print(f"Número de amostras originais: {len(df)}")
    print("Colunas do dataset:", df.columns.tolist())
    print("Exemplo das primeiras 5 linhas:\n", df.head())
    print("-" * 40)
except FileNotFoundError:
    print(f"ERRO: O arquivo '{file_path}' não foi encontrado.")
    print("Por favor, certifique-se de que você baixou 'IMBD.csv' do Kaggle")
    print("e o colocou na mesma pasta deste script Python.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar ou inspecionar o CSV: {e}")
    exit()

# Verificar e lidar com colunas nulas ou ausentes
required_columns = ['description', 'genre', 'rating', 'title']
for col in required_columns:
    if col not in df.columns:
        print(f"ERRO: Coluna '{col}' não encontrada no dataset.")
        print("Verifique se você baixou o dataset correto do Kaggle e os nomes das colunas.")
        exit()

# Remover linhas com valores nulos nas colunas que usaremos para o modelo
df.dropna(subset=['description', 'genre', 'rating', 'title'], inplace=True)
print(f"Número de amostras após remover nulos nas colunas essenciais: {len(df)}")

# --- DEFININDO SENTIMENTO COM BASE NA COLUNA 'rating' ---
# Limiar para classificação: rating >= 7.0 é considerado positivo, < 7.0 é negativo
RATING_THRESHOLD = 7.0
df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= RATING_THRESHOLD else 'negative')

# Mapear sentimentos para números (0: negative, 1: positive)
sentiment_mapping = {'negative': 0, 'positive': 1}
df['sentiment_encoded'] = df['sentiment'].map(sentiment_mapping)


df['genre'] = df['genre'].fillna('')

df['combined_text_feature'] = df['description'] + " " + df['genre'] 

# Aplicar pré-processamento de texto à nova coluna 'combined_text_feature'
print("Aplicando pré-processamento de texto à coluna combinada (descrição + gênero)...")
df['processed_combined_text_feature'] = df['combined_text_feature'].apply(preprocess_text)
print("Pré-processamento concluído.")

X_text = df['processed_combined_text_feature']
y_sentiment = df['sentiment_encoded']

# --- 2. Divisão Treino/Teste ---
print("-" * 40)
print("Dividindo dados em conjuntos de treino e teste...")
X_train_text, X_test_text, y_train, y_test, _, X_test_original_title = train_test_split(
    X_text, y_sentiment, df['title'],
    test_size=0.2, random_state=42, stratify=y_sentiment
)

print(f"Dados de Treino: {len(X_train_text)} amostras")
print(f"Dados de Teste: {len(X_test_text)} amostras")
print("-" * 40)

# --- 3. Vetorização com TF-IDF (com N-grams) ---
print("Treinando TF-IDF Vectorizer (com N-grams)...")
# Adicionado ngram_range=(1, 2) para incluir unigrams e bigrams
tfidf_vectorizer = TfidfVectorizer(max_features=10000, min_df=2, ngram_range=(1, 2))

X_train_vectorized = tfidf_vectorizer.fit_transform(X_train_text)
X_test_vectorized = tfidf_vectorizer.transform(X_test_text)

print(f"TF-IDF Vectorizer treinado. Número de features: {X_train_vectorized.shape[1]}")
print("-" * 40)

# --- 4. Treinamento dos Modelos (Regressão Logística e Naive Bayes) ---
print("Treinando Modelo de Regressão Logística...")
# Adicionado class_weight='balanced'
logistic_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced')
logistic_model.fit(X_train_vectorized, y_train)
print("Modelo de Regressão Logística treinado com sucesso!")

print("\nTreinando Modelo Multinomial Naive Bayes...")
# MultinomialNB é bom para contagens/frequências (TF-IDF)
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_vectorized, y_train)
print("Modelo Multinomial Naive Bayes treinado com sucesso!")
print("-" * 40)

# --- 5. Avaliação dos Modelos ---
print("--- Avaliação do Modelo de Regressão Logística no Conjunto de Teste ---")
y_pred_logistic = logistic_model.predict(X_test_vectorized)
y_pred_proba_logistic = logistic_model.predict_proba(X_test_vectorized)

print(f"Acurácia (Regressão Logística) no conjunto de teste: {accuracy_score(y_test, y_pred_logistic):.2f}")
print("\nRelatório de Classificação (Regressão Logística):\n", classification_report(y_test, y_pred_logistic, target_names=['negative_rating', 'positive_rating']))

fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_pred_proba_logistic[:, 1])
auc_score_logistic = auc(fpr_logistic, tpr_logistic)
print(f"AUC (Regressão Logística): {auc_score_logistic:.2f}")

# Plotar ROC para Regressão Logística
plt.figure(figsize=(8, 6))
plt.plot(fpr_logistic, tpr_logistic, color='blue', lw=2, label=f'Curva ROC Regressão Logística (AUC = {auc_score_logistic:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Classificador Aleatório')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (False Positive Rate)')
plt.ylabel('Taxa de Verdadeiros Positivos (True Positive Rate)')
plt.title('Curva ROC para Classificação de Sentimento (Regressão Logística)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print("-" * 40)
print("\n--- Avaliação do Modelo Multinomial Naive Bayes no Conjunto de Teste ---")
y_pred_nb = naive_bayes_model.predict(X_test_vectorized)
y_pred_proba_nb = naive_bayes_model.predict_proba(X_test_vectorized)

print(f"Acurácia (Naive Bayes) no conjunto de teste: {accuracy_score(y_test, y_pred_nb):.2f}")
print("\nRelatório de Classificação (Naive Bayes):\n", classification_report(y_test, y_pred_nb, target_names=['negative_rating', 'positive_rating']))

fpr_nb, tpr_nb, _ = roc_curve(y_test, y_pred_proba_nb[:, 1])
auc_score_nb = auc(fpr_nb, tpr_nb)
print(f"AUC (Naive Bayes): {auc_score_nb:.2f}")

# Plotar ROC para Naive Bayes
plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, color='red', lw=2, label=f'Curva ROC Naive Bayes (AUC = {auc_score_nb:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Classificador Aleatório')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (False Positive Rate)')
plt.ylabel('Taxa de Verdadeiros Positivos (True Positive Rate)')
plt.title('Curva ROC para Classificação de Sentimento (Multinomial Naive Bayes)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print("-" * 40)

# --- 6. Função de Classificação Interativa ---
def classificar_sentimento_interativo(model_to_use, model_name): # Recebe o modelo e seu nome
    limpar_console()
    print(f"## Classificador Probabilístico de Filmes (Baseado em Nota IMDb) - Modelo: {model_name} ##")
    print("-------------------------------------------------------------------")
    print("Este modelo prevê a probabilidade de um filme ter uma nota IMDb alta (>7.0) ou baixa (<7.0).")
    print("Você pode inserir o nome do filme, sua descrição e seus gêneros.")
    print("Ex: 'O Poderoso Chefão'")
    print("Descrição: 'Uma epopeia sobre a família Corleone, que ascende ao poder na máfia de Nova York.'")
    print("Gêneros: 'Crime, Drama, Máfia'")
    print("Ou digite 'sair' para encerrar.")
    print("-------------------------------------------------------------------")

    reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}

    while True:
        title_input = input("\nNome do Filme/Série (ou 'sair'): ").strip()
        if title_input.lower() == 'sair':
            print("Obrigado por usar o recomendador! Até mais.")
            break

        description_input = input("Descrição do Filme/Série: ").strip()
        genre_input = input("Gêneros (separados por vírgula, Ex: Ação, Drama): ").strip()

        if not description_input and not genre_input:
            print("Por favor, digite a descrição e/ou os gêneros para a análise.")
            continue

        try:
            combined_input = description_input + " " + genre_input
            processed_input = preprocess_text(combined_input)
            input_vetorized = tfidf_vectorizer.transform([processed_input])

            if input_vetorized.sum() == 0:
                print("Não consegui reconhecer nenhuma palavra da sua entrada no meu vocabulário. Tente uma descrição/gênero mais detalhada ou comum.")
                print("Lembre-se que palavras muito raras ou que não foram vistas no treinamento podem ser ignoradas.")
                continue

            probabilidades = model_to_use.predict_proba(input_vetorized)[0]
            prob_negativo_rating = probabilidades[0]
            prob_positivo_rating = probabilidades[1]

            previsao_encoded = model_to_use.predict(input_vetorized)[0]
            previsao_sentimento = reverse_sentiment_mapping[previsao_encoded]

            print("\n--- Análise Probabilística para o Filme ---")
            print(f"Filme/Série: '{title_input}'")
            print(f"Descrição: '{description_input}'")
            print(f"Gêneros Analisados: '{genre_input}'")
            print(f"Probabilidade de ter **NOTA BAIXA** (<{RATING_THRESHOLD:.1f}): {prob_negativo_rating:.2%}")
            print(f"Probabilidade de ter **NOTA ALTA** (>={RATING_THRESHOLD:.1f}): {prob_positivo_rating:.2%}")
            
            if previsao_sentimento == 'positive':
                print(f"\n**Previsão:** 🎉 **PROVAVELMENTE É UM FILME/SÉRIE BEM AVALIADO!** 🎉")
            else:
                print(f"\n**Previsão:** 😔 **PROVAVELMENTE É UM FILME/SÉRIE COM NOTA BAIXA.** 😔")

            print("\n-------------------------------------------------------------------")
            input("Pressione Enter para continuar...")
            limpar_console()
            print(f"## Classificador Probabilístico de Filmes (Baseado em Nota IMDb) - Modelo: {model_name} ##")
            print("-------------------------------------------------------------------")
            print("Digite os dados de outro filme/série ou 'sair' para encerrar.")
            print("-------------------------------------------------------------------")

        except Exception as e:
            print(f"Ocorreu um erro: {e}. Tente novamente.")
            print(f"Detalhes do erro: {e}")
            continue

# --- Iniciar a aplicação ---
if __name__ == "__main__":
    # Permite ao usuário escolher qual modelo usar
    while True:
        limpar_console()
        print("Escolha o modelo para a classificação interativa:")
        print("1. Regressão Logística")
        print("2. Multinomial Naive Bayes")
        print("3. Sair")
        choice = input("Digite o número da sua escolha: ").strip()

        if choice == '1':
            classificar_sentimento_interativo(logistic_model, "Regressão Logística")
            break # Sai do loop de escolha após o uso do classificador
        elif choice == '2':
            classificar_sentimento_interativo(naive_bayes_model, "Multinomial Naive Bayes")
            break # Sai do loop de escolha após o uso do classificador
        elif choice == '3':
            print("Encerrando o programa.")
            break
        else:
            print("Escolha inválida. Por favor, digite 1, 2 ou 3.")
            input("Pressione Enter para continuar...")