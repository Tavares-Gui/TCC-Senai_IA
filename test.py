import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

with open('chatbot_classifier_rf.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

stop_words = set(stopwords.words('portuguese'))

def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

def get_response(classifier, vectorizer, question):
    processed_question = preprocess(question)
    question_vector = vectorizer.transform([processed_question])
    return classifier.predict(question_vector)[0]

example_questions = [
    "Quem gerencia o benefício Transporte Fretado?",
    "Quem pode usar o Transporte Fretado?",
    "Quando o usuário Bosch cadastrado não poderá usar o Transporte Fretado?"
]

for question in example_questions:
    response = get_response(clf, vectorizer, question)
    print(f"Pergunta: {question}")
    print(f"Resposta: {response}")
    print()

while True:
    user_input = input("Você: ")
    if user_input.lower() == "sair":
        break
    response = get_response(clf, vectorizer, user_input)
    print("Bot:", response)
 