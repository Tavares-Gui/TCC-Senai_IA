import nltk
import string
import pandas as pd
import matplotlib.pyplot as plt

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

file_path = './PerguntasRespostas.csv'
data = pd.read_csv(file_path)

remove_chars = "?.,;!:"

def clean_text(text):
    return ''.join(char for char in text if char not in remove_chars)

all_questions = []
perguntas = data['Perguntas']

all_answers = []
respostas = data['Respostas']

for text in perguntas:
    cleaned_questions = clean_text(text)
    questions = word_tokenize(cleaned_questions)
    all_questions.extend(questions)

for text in respostas:
    answers = word_tokenize(text)
    all_answers.extend(answers)

fdq = FreqDist(all_questions)
fda = FreqDist(all_answers)

print(fdq.most_common(10))
print(fda.most_common(10))

fdq.plot(30, cumulative=False)
fda.plot(30, cumulative=False)
plt.show()
