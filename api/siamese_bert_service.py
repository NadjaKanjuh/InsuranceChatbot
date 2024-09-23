import os
import torch
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, models


class BertForSTS(torch.nn.Module):
    def __init__(self):
        super(BertForSTS, self).__init__()
        self.bert = models.Transformer('distilbert-base-uncased', max_seq_length=64)
        self.pooling_layer = models.Pooling(self.bert.get_word_embedding_dimension())
        self.sts_bert = SentenceTransformer(modules=[self.bert, self.pooling_layer])

    def forward(self, input_data):
        output = self.sts_bert(input_data)['sentence_embedding']
        return output


class HighPrecisionServiceBert:
    def __init__(self):
        self.current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.current_file_dir)
        self.model_weights = os.path.join(self.base_dir, 'bert_model', 'bert_model_weights.pth')
        self.tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = BertForSTS()
        self.model.load_state_dict(torch.load(self.model_weights, map_location='cpu'))
        self.model.eval()

    def tokenize_questions(self, question1, question2):
        return self.tokenizer([question1, question2], padding='max_length', max_length=128, truncation=True,
                         return_tensors="pt")


    def predict_similarity(self, question1, question2):
        tokenized_questions = self.tokenize_questions(question1, question2)
        tokenized_questions['input_ids'] = tokenized_questions['input_ids']
        tokenized_questions['attention_mask'] = tokenized_questions['attention_mask']
        del tokenized_questions['token_type_ids']
        output = self.model(tokenized_questions)
        embeddings1 = output[0]
        embeddings2 = output[1]
        sim = torch.nn.functional.cosine_similarity(embeddings1.unsqueeze(0), embeddings2.unsqueeze(0)).item()

        return sim


    def get_most_similar_question(self, input_question, question_list):
        # Find the most similar question from a list of questions
        similarities = []
        for question in question_list:
            similarity = self.predict_similarity(input_question, question)
            similarities.append(similarity)

        # Get the question with the highest similarity score
        max_sim_index = torch.argmax(torch.tensor(similarities))
        return question_list[max_sim_index]


