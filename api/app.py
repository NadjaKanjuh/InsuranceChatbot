from flask import Flask, request
import pandas as pd
import string


app = Flask(__name__)

# Flag to track if the initialization has already been done
initialized = False

@app.before_request
def initialize():
    global initialized
    if not initialized:
        import high_recall_service
        import siamese_bert_service
        global high_recall_service_obj, high_precision_service_obj, siamese_bert_service_obj, qa_dict
        high_recall_service_obj = high_recall_service.HighRecallService()
        siamese_bert_service_obj = siamese_bert_service.HighPrecisionServiceBert()

        df = pd.read_csv('../data/insurance_qna_dataset.csv', sep='\t')
        df['Question'] = df['Question'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        df['Question'] = df['Question'].apply(lambda x: x.lower())
        qa_dict = dict(zip(df['Question'], df['Answer']))

        initialized = True

@app.route('/get-answer', methods=['POST'])
def get_answer():
    if request.method == 'POST':
        user_question = request.get_data().decode('utf-8')  # Ensure correct decoding
        questions = high_recall_service_obj.get_N_similar_questions(user_question)  # Get the top 100 most similar questions

        most_similar_question = siamese_bert_service_obj.get_most_similar_question(user_question, questions)

        print("Most similar question:", most_similar_question)

        answer = qa_dict.get(most_similar_question, "I'm sorry, I don't have an answer for that.")

        return answer
    else:
        return 'Invalid request method'

if __name__ == '__main__':
    app.run(debug=True)
