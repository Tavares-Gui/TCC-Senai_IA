import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

file_path = './csv/PerguntasRespostasBig.csv'
data = pd.read_csv(file_path)

stop_words = set(stopwords.words('portuguese'))

def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

data['Perguntas'] = data['Perguntas'].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Perguntas'])
y = data['Respostas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 7]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

with open('chatbot_classifier_rf.pkl', 'wb') as model_file:
    pickle.dump(best_rf, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Modelo treinado salvo como 'chatbot_classifier_rf.pkl'.")
print("Vectorizador TF-IDF salvo como 'tfidf_vectorizer.pkl'.")
