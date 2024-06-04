from flask import Flask, request
import pandas as pd
import string

app = Flask(__name__)

@app.before_first_request
def load():
    import high_recall_service
    import high_precision_service
    global high_recall_service_obj, high_precision_service_obj, qa_dict
    high_recall_service_obj = high_recall_service.HighRecallService()
    high_precision_service_obj = high_precision_service.HighPrecisionService()

    df = pd.read_csv('../data/insurance_qna_dataset.csv', sep='\t')
    df['Question'] = df['Question'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['Question'] = df['Question'].apply(lambda x: x.lower())
    qa_dict = dict(zip(df['Question'], df['Answer']))


@app.route('/get-answer', methods=['POST'])
def get_answer():
    if request.method == 'POST':
        user_question = request.get_data()
        questions = high_recall_service_obj.get_N_similar_questions(user_question) #vracam najslicnijih 100 pitanja

        most_similar_question = high_precision_service_obj.get_most_similar_question(questions, user_question)

        print("Most similar question:", most_similar_question)

        answer = qa_dict.get(most_similar_question, "I'm sorry, I don't have an answer for that.")

        #vracam odgovor
        return answer
    else:
        return 'Invalid request method'

if __name__ == '__main__':
    app.run()