import pandas as pd
import nltk
import string
import numpy as np
import gensim.downloader
from sklearn.neighbors import NearestNeighbors


class HighRecallService:
    def __init__(self):
        print("High recall service")
        self.corpus_full = None
        self.corpus_train = None
        self.corpus_tokenized = None
        self.translator = None

        self.vectors = gensim.downloader.load('word2vec-google-news-300')

        self.document_vector_sum = []

        self.model = None

        self.__load_data()

    def __load_data(self):
        corpus = pd.read_csv('../data/insurance_qna_dataset.csv', sep='\t')
        corpus_question_list = corpus['Question'].drop_duplicates().tolist()
        corpus_answers_list = corpus['Answer'].drop_duplicates().tolist()
        self.corpus_full = corpus_question_list + corpus_answers_list
        self.corpus_train = corpus_question_list

        self.__tokenize_corpus()
        self.__get_document_vector_sum()
        self.__train_knn()

    def __tokenize_corpus(self):
        self.translator = str.maketrans('', '', string.punctuation)
        self.corpus_train = [doc.translate(self.translator).lower() for doc in self.corpus_train]
        self.corpus_tokenized = [nltk.word_tokenize(doc) for doc in self.corpus_train]

    def __get_document_vector_sum(self):
        for doc in self.corpus_tokenized:
            word_vectors = [self.vectors[word] for word in doc if word in self.vectors.key_to_index]
            if len(word_vectors) > 0:
                document_vector = np.sum(word_vectors, axis=0)
            else:
                document_vector = np.zeros(self.vectors.vector_size)
            self.document_vector_sum.append(document_vector)

    def __train_knn(self):
        self.model = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='euclidean').fit(self.document_vector_sum)

    def get_N_similar_questions(self, question):
        # Convert input to string
        question_str = str(question)

        # Tokenize input question
        question_tokenized = nltk.word_tokenize(question_str.lower())

        # Vectorize input question
        question_vector = np.sum([self.vectors[word] for word in question_tokenized if word in self.vectors.key_to_index], axis=0)

        # Find 100 most similar questions and return them
        _, indices = self.model.kneighbors([question_vector])
        similar_questions = [self.corpus_train[idx] for idx in indices[0]]

        return similar_questions
