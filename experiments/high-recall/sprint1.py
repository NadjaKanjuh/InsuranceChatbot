import nltk
import pandas as pd
import numpy as np
import spacy
#nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from data.test_data import original, easy, medium, hard
from sklearn.neighbors import NearestNeighbors
from nltk.stem import SnowballStemmer


corpus_train = pd.read_csv('../../data/insurance_qna_dataset.csv', sep='\t')
corpus_question = corpus_train['Question']
corpus_question = corpus_question.drop_duplicates()
corpus_answers = corpus_train['Answer']
corpus_answers = corpus_answers.drop_duplicates()
corpus = pd.concat([corpus_question, corpus_answers], axis=1, join='inner')
corpus_full = list(corpus_question) + list(corpus_answers)
corpus_train = list(corpus_question)
changed = easy + medium + hard
corpus_test = pd.DataFrame({'Original': 3 * original, 'Changed': changed})


class Lemmatization:
    def __init__(self):
        self.lem = spacy.load('en_core_web_sm')

    def lemmatize_text(self, text):
        doc = self.lem(text)
        lemmatized_text = ' '.join([token.lemma_ for token in doc])
        return lemmatized_text

    def lemmatize_dataframe(self, df, column):
        df_lemmatized = df.copy()
        df_lemmatized[column] = df_lemmatized[column].apply(self.lemmatize_text)
        return df_lemmatized

class Stemming():
    def __init__(self):
        self.stemmer = SnowballStemmer(language='english')

    def stem_text(self, text):
        doc = nltk.word_tokenize(text)
        stemmed_text = ' '.join([self.stemmer.stem(token) for token in doc])
        return  stemmed_text

    def stem_dataframe(self, df, column):
        df_stemmed = df.copy()
        df_stemmed[column] = df_stemmed[column].apply(self.stem_text)
        return df_stemmed

stemmer = Stemming()
corpus_stemmed = stemmer.stem_dataframe(corpus, 'Question')
corpus_stemmed_full = list(corpus_stemmed["Question"])+list(corpus_stemmed["Answer"])
corpus_stemmed_questions = list(corpus_stemmed["Question"])
corpus_stemmed_test = stemmer.stem_dataframe(corpus_test, 'Changed')



#lemmatizer = Lemmatization()
#corpus_lemmatized = lemmatizer.lemmatize_dataframe(corpus, 'Question')
#corpus_lemmatized_full = list(corpus_lemmatized["Question"])+list(corpus_lemmatized["Answer"])
#corpus_lemmatized_questions = list(corpus_lemmatized["Question"])
#corpus_lemmatized_test = lemmatizer.lemmatize_dataframe(corpus_test, 'Changed')

class TfIdfVectorizerClass:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2))

    def fit(self, corpus):
        return self.vectorizer.fit(corpus)

    def transform(self, corpus):
        return self.vectorizer.transform(corpus)


tfidf_vectorizer = TfIdfVectorizerClass()
tfidf_vectorizer.fit(corpus_full)
tfidf_matrix = tfidf_vectorizer.transform(corpus_train)
vectorized = tfidf_vectorizer.transform(corpus_test["Changed"])
nbrs_euclidean = NearestNeighbors(n_neighbors=100, metric='euclidean').fit(tfidf_matrix)
nbrs_manhattan = NearestNeighbors(n_neighbors=100, metric='manhattan').fit(tfidf_matrix)
nbrs_cosine = NearestNeighbors(n_neighbors=100, metric='cosine').fit(tfidf_matrix)

def calculate_rank(metric, matrix, corpus_train):
    if metric == 'euclidean':
        distances, indices = nbrs_euclidean.kneighbors(matrix)
        #print("DE", distances) #vraca sortirane distance
        #print(indices)
    elif metric == 'manhattan':
        distances, indices = nbrs_manhattan.kneighbors(matrix)
        #print("MH", distances)
    else:
        distances, indices = nbrs_cosine.kneighbors(matrix)
        #print("CO", distances)



    ranks = []
    for index_list in indices:
        for i, index in enumerate(index_list):
            #print(corpus_train.index(corpus_train[index]), index)
            if corpus_train.index(corpus_train[index]) == index:
                ranks.append(i)
            else:
                ranks.append(200)
    return ranks


def calculate_mean_rank(matrix, corpus_train):
    euclidean_rank = calculate_rank('euclidean', matrix, corpus_train)
    manhattan_rank = calculate_rank('manhattan', matrix, corpus_train)
    cosine_rank = calculate_rank('cosine', matrix, corpus_train)

    mean_euclidean_rank = np.mean(euclidean_rank)
    mean_manhattan_rank = np.mean(manhattan_rank)
    mean_cosine_rank = np.mean(cosine_rank)

    return mean_euclidean_rank, mean_manhattan_rank, mean_cosine_rank


def find_best_metric(matrix, corpus_train):
    mean_euclidean_rank, mean_manhattan_rank, mean_cosine_rank = calculate_mean_rank(matrix, corpus_train)
    best_rank = min(mean_euclidean_rank, mean_manhattan_rank, mean_cosine_rank)

    if best_rank == mean_euclidean_rank:
        best_metric = 'euclidean'
    elif best_rank == mean_manhattan_rank:
        best_metric = 'manhattan'
    else:
        best_metric = 'cosine'

    return print("The best metric is", best_metric, ".Rank is: ", best_rank)

print("Mean ranks - euclidean, manhattan & cosine", calculate_mean_rank(vectorized, corpus))
print(find_best_metric(vectorized, corpus_train))





