import gensim.downloader
import pandas as pd
import numpy as np
import re
import gensim.downloader
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import nltk
nltk.download('punkt')
import tensorflow as tf

from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Embedding, Lambda
import tensorflow.python.keras.backend as K
# from keras.src.layers import LSTM
from tensorflow.python.keras.layers.recurrent import LSTM





def text_to_word_list(text):
    # Preprocess and convert texts to a list of words
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


class HighPrecisionService:
    def __init__(self):
        print("High precision service")
        self.df = None
        self.train_df = None
        self.test_df = None

        print("Loading word2vec model")
        self.word2vec = gensim.downloader.load('word2vec-google-news-300')
        self.vocabulary = None
        self.inverse_vocabulary = None
        self.stops = None

        self.questions_cols = ["question1", "question2"]

        self.embedding_dim = self.word2vec.vector_size
        self.max_seq_length = None
        self.embeddings = None

        self.model = None

        self.__load_data()

    def __load_data(self):
        self.df = pd.read_csv('../data/quora_duplicate_questions.tsv', sep='\t')
        self.train_df, self.test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.test_df = self.test_df.drop('is_duplicate', axis=1)

        self.stops = set(stopwords.words('english'))

        self.__build_vocabulary()
        self.max_seq_length = 40

        self.__build_and_load_model()

    def __build_vocabulary(self):
        self.vocabulary = dict()
        self.inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding

        for dataset in [self.train_df, self.test_df]:
            for index, row in dataset.iterrows():

                # Iterate through the text of both questions of the row
                for question in self.questions_cols:

                    word_indices = []  # word_indices -> question numbers representation
                    for word in text_to_word_list(row[question]):

                        # Check for unwanted words
                        if word in self.stops and word not in self.word2vec.key_to_index:
                            continue

                        if word not in self.vocabulary:
                            self.vocabulary[word] = len(self.inverse_vocabulary)
                            word_indices.append(len(self.inverse_vocabulary))
                            self.inverse_vocabulary.append(word)
                        else:
                            word_indices.append(self.vocabulary[word])

                    # Replace questions as word to question as number representation
                    dataset.at[index, question] = word_indices

    def __exponent_neg_manhattan_distance(self, left, right):
        # ''' Helper function for the similarity estimate of the LSTMs outputs'''
        return tf.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

    def __build_and_load_model(self):
        n_hidden = 50
        gradient_clipping_norm = 1.25
        self.embeddings = 1 * np.random.randn(len(self.vocabulary) + 1, self.embedding_dim) # This will be the embedding matrix
        self.embeddings[0] = 0

        # The visible layer
        left_input = Input(shape=(self.max_seq_length,), dtype='int32')
        right_input = Input(shape=(self.max_seq_length,), dtype='int32')

        embedding_layer = Embedding(len(self.embeddings), self.embedding_dim, weights=[self.embeddings], input_length=self.max_seq_length,
                                    trainable=False)

        # Embedded version of the inputs
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        # Since this is a siamese network, both sides share the same LSTM
        shared_lstm = LSTM(n_hidden)

        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # Calculates the distance as defined by the MaLSTM model
        malstm_distance = Lambda(function=lambda x: self.__exponent_neg_manhattan_distance(x[0], x[1]),
                                 output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        # Pack it all up into a model
        malstm = Model([left_input, right_input], [malstm_distance])

        # Adadelta optimizer, with gradient clipping by norm
        # optimizer = Adadelta(clipnorm=gradient_clipping_norm)

        malstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        malstm.load_weights("../experiments/high-precision/malstm_trained_weights.h5")
        self.model = malstm

        return malstm

    def __question_to_sequence(self, question):
        q2n = [] #question to number representation

        for word in text_to_word_list(question):
            if word in self.stops and word not in self.word2vec.key_to_index:
                continue

            if word in self.vocabulary:
                q2n.append(self.vocabulary[word])
            else:
                q2n.append(0)

        return q2n

    def get_most_similar_question(self, questions, question):
        # Repeat the question for comparison
        asked_question_repeated = np.repeat([question], len(questions), axis=0)

        # Convert both questions and question list to sequences
        asked_question_sequence = [self.__question_to_sequence(q) for q in asked_question_repeated]
        questions_sequence = [self.__question_to_sequence(q) for q in questions]

        # Ensure sequences are the same length
        asked_question_sequence = pad_sequences(asked_question_sequence, maxlen=self.max_seq_length)
        questions_sequence = pad_sequences(questions_sequence, maxlen=self.max_seq_length)

        # Make sure to cast the inputs to numpy arrays or tensors
        asked_question_sequence = np.array(asked_question_sequence)
        questions_sequence = np.array(questions_sequence)

        # Predictions
        predictions = self.model.predict([asked_question_sequence, questions_sequence])

        # Flatten the predictions and find the most similar question
        predictions_flattened = predictions.flatten()
        top_prediction_index = np.argmax(predictions_flattened)  # Get the index of the highest score
        top_prediction = questions[top_prediction_index]
        return top_prediction
