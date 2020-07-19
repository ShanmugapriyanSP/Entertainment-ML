import pickle
import tensorflow as tf
import nltk

nltk.download('stopwords')

from tensorflow.keras.models import load_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

tokenizer = pickle.load(open('entertainment_ml/files/tokenizer.pkl', 'rb'))
model = load_model('entertainment_ml/files/model_LSTMFullData.h5')
