from django.shortcuts import render

import nltk
import re
import traceback
import numpy as np

from . import tokenizer, model

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_review_type(review_test):
    review_test = re.sub('[^a-zA-Z]', ' ', review_test)
    review_test = review_test.lower()
    review_test = review_test.split()
    ps = PorterStemmer()
    review_test = [ps.stem(word) for word in review_test if not word in set(stopwords.words('english'))]
    review_test = ' '.join(review_test)
    review_test = [review_test]
    seq_test = tokenizer.texts_to_sequences(review_test)
    data_test = pad_sequences(seq_test, maxlen=921)
    result = model.predict(data_test)[0]

    labels = ['Review is Negative',
              'Review is Somewhat Negative',
              'Review is Neutral',
              'Review is Somewhat Positive',
              'Review is Positive'
              ]
    result = labels[np.argmax(result)]
    return result


def home(request):
    context = {
        'title': 'Sentiment Analysis'
    }
    return render(request, 'entertainment_ml/home.html', context)


def predict(request):
    try:
        test_review = request.POST['review']
        if test_review == '':
            result = ''
        else:
            result = predict_review_type(test_review)
        context = {
            'title': 'Sentiment Result',
            'result': result
        }
    except Exception as exc:
        print('Exception - %s', exc)
        print(traceback.format_exc())
        context = {
            'title': 'Error'
        }
        return render(request, 'entertainment_ml/home.html', context)

    return render(request, 'entertainment_ml/home.html', context)
