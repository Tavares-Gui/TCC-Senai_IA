# pip install Flask scikit-learn nltk

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')

from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

with open('../chatbot_classifier_rf.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

with open('../tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
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

@app.route('/perguntar', methods=['POST'])
def perguntar():
    data = request.get_json()
    pergunta = data.get('pergunta', '')
    if not pergunta:
        return jsonify({"erro": "Nenhuma pergunta fornecida."}), 400
    
    resposta = get_response(clf, vectorizer, pergunta)
    return jsonify({"pergunta": pergunta, "resposta": resposta})

if __name__ == '__main__':
    app.run(debug=True)
