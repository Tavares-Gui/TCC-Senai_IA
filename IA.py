import nltk
import string
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, RSLPStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

file_path = './csv/PerguntasRespostasBig.csv'
data = pd.read_csv(file_path)

remove_chars = "?.;!:"
def clean_text(text):
    return ''.join(char for char in text if char not in remove_chars)

stop_words = set(stopwords.words('portuguese'))
def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    return sia.polarity_scores(text)

all_questions = []
perguntas = data['Perguntas']

all_answers = []
respostas = data['Respostas']

apply_lemmatization = True

for text in perguntas:
    cleaned_questions = clean_text(text)
    questions = word_tokenize(cleaned_questions)
    questions = remove_stopwords(questions)
    if apply_lemmatization:
        questions = lemmatize_tokens(questions)
    all_questions.extend(questions)

fdq = FreqDist(all_questions)

print(fdq.most_common(10))

for i, text in enumerate(perguntas):
    sentiment = analyze_sentiment(text)
    print(f"Sentimento da Pergunta {i+1}: {sentiment}")
