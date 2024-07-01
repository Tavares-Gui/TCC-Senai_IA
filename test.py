import nltk
import pandas as pd
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Carregar o classificador treinado a partir do arquivo .pkl
model_path = 'chatbot_classifier.pkl'

with open(model_path, 'rb') as model_file:
    classifier = pickle.load(model_file)

# Função para pré-processar a pergunta
stop_words = set(stopwords.words('portuguese'))

def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens

# Função para prever a resposta a uma nova pergunta
def get_response(classifier, question):
    processed_question = preprocess(question)
    features = dict([(token, True) for token in processed_question])
    return classifier.classify(features)

# Exemplo de uso
example_questions = [
    "Quem gerencia o benefício Transporte Fretado?",
    "Quem pode usar o Transporte Fretado?",
    "Quando o usuário Bosch cadastrado não poderá usar o Transporte Fretado?"
]

for question in example_questions:
    response = get_response(classifier, question)
    print(f"Pergunta: {question}")
    print(f"Resposta: {response}")
    print()

# Adicione uma interface simples para entrada do usuário
while True:
    user_input = input("Você: ")
    if user_input.lower() == "sair":
        break
    response = get_response(classifier, user_input)
    print("Bot:", response)
