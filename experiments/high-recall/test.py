
import pandas as pd


corpus_train = pd.read_csv('../../data/insurance_qna_dataset.csv', sep='\t')
print("prvo pitanje", corpus_train['Question'][0])
print("prvo odg", corpus_train['Answer'][0])
print("drugo pitanje", corpus_train['Question'][3])
print("drugi odg", corpus_train['Answer'][3])
print("trece pitanje", corpus_train['Question'][6])
print("treci odg", corpus_train['Answer'][6])

corpus_question = corpus_train['Question']
corpus_question = corpus_question.drop_duplicates()
corpus_answers = corpus_train['Answer']
corpus_answers = corpus_answers.drop_duplicates()
corpus = pd.concat([corpus_question, corpus_answers], axis=1, join='inner')
corpus_full = list(corpus_question) + list(corpus_answers)

print(len(corpus_full))