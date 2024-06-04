import pandas as pd
import nltk
import string
import numpy as np
import spacy
from data.test_data import original, easy, medium, hard
from sklearn.neighbors import NearestNeighbors
import gensim.downloader
from sklearn.feature_extraction.text import TfidfVectorizer


corpus_train = pd.read_csv('../../data/insurance_qna_dataset.csv', sep='\t')
corpus_question_list = corpus_train['Question'].drop_duplicates().tolist()
corpus_answers_list = corpus_train['Answer'].drop_duplicates().tolist()
corpus_full = corpus_question_list + corpus_answers_list
corpus_train = corpus_question_list
changed = easy + medium + hard
corpus_test = pd.DataFrame({'Original': 3 * original, 'Changed': changed})
corpus_test_question = corpus_test['Original'].drop_duplicates().tolist()
corpus_test_changed = corpus_test['Changed'].tolist()
corpus_test = corpus_test_question + corpus_test_changed
#print(corpus_test_changed)

translator = str.maketrans('', '', string.punctuation)
corpus_train = [doc.translate(translator).lower() for doc in corpus_train]
corpus_tokenized = [nltk.word_tokenize(doc) for doc in corpus_train] #tokenizacija pitanja
#print(corpus_tokenized[0])

corpus_test_changed = [doc.translate(translator).lower() for doc in corpus_test_changed]
test_tokenized = [nltk.word_tokenize(doc) for doc in corpus_test_changed] #tokenizacija test skupa
#print(test_tokenized[0])


#print(list(gensim.downloader.info()['models'].keys()))

vectors = gensim.downloader.load('word2vec-google-news-300')
#print(vectors.most_similar('insurance')) #insurers, insurance_premiums, insurance, premiums, insurer, insured, insurances, reinsurance,

def calculate_document_vectors(corpus_tokenized, vectors, method='average'):
    document_vectors = []
    for doc in corpus_tokenized:
        word_vectors = [vectors[word] for word in doc if word in vectors.vocab]
        if len(word_vectors) > 0:
            if method == 'average':
                document_vector = np.mean(word_vectors, axis=0)
            elif method == 'sum':
                document_vector = np.sum(word_vectors, axis=0)
            else:
                raise ValueError('Invalid method. Supported methods are "average" and "sum".')
        else:
            document_vector = np.zeros(vectors.vector_size)
        document_vectors.append(document_vector)

    return document_vectors

#vektor dokumenta za trening skup
document_vector_average = calculate_document_vectors(corpus_tokenized, vectors, 'average')
#print(document_vector_average[0])
document_vector_sum = calculate_document_vectors(corpus_tokenized, vectors, 'sum')
#print(document_vector_sum[0])
#vektor dokumenta za test skup
test_vector_average = calculate_document_vectors(test_tokenized, vectors, 'average')
#print(test_vector_average[0])
test_vector_sum = calculate_document_vectors(test_tokenized, vectors, 'sum')
#print(test_vector_sum[0])


#pretrained word vectors combined with IDF weighted average and sum vector arithmetics


tfidf_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2))
tfidf_vectorizer.fit_transform(corpus_full)
idf_weights = tfidf_vectorizer.idf_


def calculate_document_vectors_idf(corpus_tokenized, vectors, tfidf_vectorizer, idf_weights, method='average'):
    document_vectors_idf = []
    for doc in corpus_tokenized:
        word_vectors = []
        for word in doc:
            if word in vectors.vocab:
                if word in tfidf_vectorizer.vocabulary_:
                    idf_weight = idf_weights[tfidf_vectorizer.vocabulary_[word]]
                else:
                    idf_weight = 1.0
                word_vectors.append(vectors[word] * idf_weight)
        if len(word_vectors) > 0:
            if method == 'average':
                document_vector = np.mean(word_vectors, axis=0)
            elif method == 'sum':
                document_vector = np.sum(word_vectors, axis=0)
            else:
                raise ValueError('Invalid method. Supported methods are "average" and "sum".')
        else:
            document_vector = np.zeros(vectors.vector_size)
        document_vectors_idf.append(document_vector)

    return document_vectors_idf

document_vectors_avg_idf = calculate_document_vectors_idf(corpus_tokenized, vectors, tfidf_vectorizer, idf_weights, 'average')
#print(document_vectors_avg_idf)
document_vectors_sum_idf = calculate_document_vectors_idf(corpus_tokenized, vectors, tfidf_vectorizer, idf_weights, 'sum')
#print(document_vectors_sum_idf)
document_vectors_avg_idf_test = calculate_document_vectors_idf(test_tokenized, vectors, tfidf_vectorizer, idf_weights, 'average')
#print(document_vectors_avg_idf_test)
document_vectors_sum_idf_test = calculate_document_vectors_idf(test_tokenized, vectors, tfidf_vectorizer, idf_weights, 'sum')
#print(document_vectors_sum_idf_test)


#Pretrained word vectors combined with POS weighted average and sum vector arithmetic

p = spacy.load("en_core_web_sm")

pos_weights = {
    "ADJ": 6.0,
    "VERB": 5.5,
    "NOUN": 7.0,
    "ADV": 3.0,
}

def get_pos_weight(pos):
    return pos_weights.get(pos, 0.0)


def calculate_document_vectors_pos(corpus, p, method='average'):
    document_vectors_pos = []

    for doc in corpus:
        word_vectors = []
        pos_tags = p(doc)
        for token in pos_tags:
            if token.has_vector:
                pos_weight = get_pos_weight(token.pos_)
                word_vectors.append(token.vector * pos_weight)

        if len(word_vectors) > 0:
            if method == 'average':
                document_vector_pos = np.mean(word_vectors, axis=0)
            elif method == 'sum':
                document_vector_pos = np.sum(word_vectors, axis=0)
            else:
                raise ValueError('Invalid method. Supported methods are "average" and "sum".')
        else:
            document_vector_pos = np.zeros(p.vocab.vectors_length)

        document_vectors_pos.append(document_vector_pos)

    return document_vectors_pos



document_vectors_avg_pos = calculate_document_vectors_pos(corpus_train, p, 'average')
#print(document_vectors_avg_pos)
document_vectors_sum_pos = calculate_document_vectors_pos(corpus_train, p, 'sum')
#print(document_vectors_sum_pos)
document_vectors_avg_pos_test = calculate_document_vectors_pos(corpus_test_changed, p, 'average')
#print(document_vectors_avg_pos_test)
document_vectors_sum_pos_test = calculate_document_vectors_pos(corpus_test_changed, p, 'sum')
#print(document_vectors_sum_pos_test)



def k_nearest(model, matrix):
    distance, indices = model.kneighbors(matrix.reshape(1,-1))
    return distance,indices

def train_knn(train_set,metric):
    if metric == 'cosine':
        nbrs = NearestNeighbors(n_neighbors=100,algorithm='brute',metric='cosine').fit(train_set)
    elif metric == 'manhattan':
        nbrs = NearestNeighbors(n_neighbors=100,algorithm='brute', metric='manhattan').fit(train_set)
    elif metric == 'euclidean':
        nbrs = NearestNeighbors(n_neighbors=100,algorithm='brute',metric='euclidean').fit(train_set)
    return nbrs


def ranking(model, test_set, corpus):
    total_rank = 0
    for elem in test_set:
        changed_question, original = elem[0], elem[1]
        _, indices = k_nearest(model, changed_question)
        results = corpus[indices[0]]
        for i, question in enumerate(results):
            if original == question:
                total_rank += i
                break
        if (i == 99):
            total_rank += 200

    avg_rank = total_rank / len(test_set)

    return avg_rank


#REZULTATI
print('-------------------------------------------------------------------')
print("Pretrained word vector with average vector arithmetics")
model_cosine_avg = train_knn(document_vector_average, 'cosine')
model_euclidean_avg = train_knn(document_vector_average, 'euclidean')
model_manhattan_avg = train_knn(document_vector_average, 'manhattan')


test_data = list(zip(test_vector_average,3*original))

print('Cosine AVG: ',ranking(model_cosine_avg,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Euclidean AVG: ',ranking(model_euclidean_avg,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Manhattan AVG: ',ranking(model_manhattan_avg,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('-------------------------------------------------------------------')

print("Pretrained word vector with sum vector arithmetics")
model_cosine_sum = train_knn(document_vector_sum, 'cosine')
model_euclidean_sum = train_knn(document_vector_sum, 'euclidean')
model_manhattan_sum = train_knn(document_vector_sum, 'manhattan')


test_data = list(zip(test_vector_sum,3*original))

print('Cosine SUM: ',ranking(model_cosine_sum,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Euclidean SUM: ',ranking(model_euclidean_sum,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Manhattan SUM: ',ranking(model_manhattan_sum,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('-------------------------------------------------------------------')

print("Pretrained word vector combined with IDF weighted average vector arithmetics")
model_cosine_avg_idf = train_knn(document_vectors_avg_idf, 'cosine')
model_euclidean_avg_idf = train_knn(document_vectors_avg_idf, 'euclidean')
model_manhattan_avg_idf = train_knn(document_vectors_avg_idf, 'manhattan')


test_data = list(zip(document_vectors_avg_idf_test,3*original))

print('Cosine AVG + IDF: ',ranking(model_cosine_avg_idf,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Euclidean AVG + IDF: ',ranking(model_euclidean_avg_idf,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Manhattan AVG + IDF: ',ranking(model_manhattan_avg_idf,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('-------------------------------------------------------------------')

print("Pretrained word vector combined with IDF weighted sum vector arithmetics")

model_cosine_sum_idf = train_knn(document_vectors_sum_idf, 'cosine')
model_euclidean_sum_idf = train_knn(document_vectors_sum_idf, 'euclidean')
model_manhattan_sum_idf = train_knn(document_vectors_sum_idf, 'manhattan')


test_data = list(zip(document_vectors_sum_idf_test,3*original))

print('Cosine SUM + IDF: ',ranking(model_cosine_sum_idf,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Euclidean SUM + IDF: ',ranking(model_euclidean_sum_idf,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Manhattan SUM + IDF: ',ranking(model_manhattan_sum_idf,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('-------------------------------------------------------------------')

print("Pretrained word vector combined with POS weighted average vector arithmetics")
model_cosine_avg_pos = train_knn(document_vectors_avg_pos, 'cosine')
model_euclidean_avg_pos = train_knn(document_vectors_avg_pos, 'euclidean')
model_manhattan_avg_pos = train_knn(document_vectors_avg_pos, 'manhattan')


test_data = list(zip(document_vectors_avg_pos_test,3*original))

print('Cosine AVG + POS: ',ranking(model_cosine_avg_pos,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Euclidean AVG + POS: ',ranking(model_euclidean_avg_pos,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Manhattan AVG + POS: ',ranking(model_manhattan_avg_pos,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('-------------------------------------------------------------------')

print("Pretrained word vector combined with POS weighted sum vector arithmetics")
model_cosine_sum_pos = train_knn(document_vectors_sum_pos, 'cosine')
model_euclidean_sum_pos = train_knn(document_vectors_sum_pos, 'euclidean')
model_manhattan_sum_pos = train_knn(document_vectors_sum_pos, 'manhattan')


test_data = list(zip(document_vectors_sum_pos_test,3*original))

print('Cosine SUM + POS: ',ranking(model_cosine_sum_pos,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Euclidean SUM + POS: ',ranking(model_euclidean_sum_pos,test_data,np.array(corpus_question_list + corpus_answers_list)))
print('Manhattan SUM + POS: ',ranking(model_manhattan_sum_pos,test_data,np.array(corpus_question_list + corpus_answers_list)))





