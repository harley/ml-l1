import os
import pickle
import re

from flask import Flask, request, jsonify

from nltk.stem import PorterStemmer

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# # Unpickle the trained classifier and write preprocessor method used
def tokenizer(text):
    return text.split(' ')

def preprocessor(text):
    # Return a cleaned version of text

    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Save emoticons for later appending
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))

    return text

# Uncomment this line after you trained your model and copied it to the same folder with app.py
tweet_classifier = pickle.load(open('../data/logisticRegression.pkl', 'rb'))

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return app.send_static_file('html/index.html')


@app.route('/classify', methods=['POST'])
def classify():
    text = request.form.get('text', None)
    assert text is not None

    # Eg: I love to hate you
    result = tweet_classifier.predict_proba(tokenizer(text))
    # [[0.9, 0.1], [0.8, 0,2], [0.7, 0.3]]
    pos_neg_list = [[max(a, b), 'Positive' if b > a else 'Negative'] for a,b in result]

    # Eg: [[0.6441255900864357, 'Positive'], [0.9887743499814371, 'Positive'], [0.6441255900864357, 'Positive'], [0.9844300054417668, 'Negative'], [0.6441255900864357, 'Positive']]
    positive_count = sum(e[1] == 'Positive' for e in pos_neg_list)

    if positive_count * 2 > len(pos_neg_list):
        # more positive -> choose largest positive prob
        s = 'Positive'
    else:
        s = 'Negative'
        # more negative -> choose largest negative prob

    p = max(e[0] for e in pos_neg_list if e[1] == s)

    # import pdb; pdb.set_trace()
    # [prob_neg, prob_pos] = result[0]

    # print('prob_neg', prob_neg)
    # print('prob_pos', prob_pos)

    # s = 'Positive' if prob_pos >= prob_neg else 'Negative'
    # p = prob_pos if prob_pos >= prob_neg else prob_neg

    return jsonify({
        'sentiment': s,
        'probability': p
    })

app.run()
