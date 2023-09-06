import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

import tensorflow
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("all")

voc_size=18611

lemmatizer=WordNetLemmatizer()
def lemmitization(sentence):
  sentence=re.sub('[^a-zA-Z]', ' ',sentence)
  sentence=sentence.lower()
  sentence=sentence.split()
  sentence=[lemmatizer.lemmatize(word) for word in sentence if not word in set(stopwords.words("english"))]
  sentence=" ".join(sentence)
  return sentence

def embedded_operation(data):
  onehot_rep=[one_hot(words,voc_size)for words in data]
  maximum=1844
  for rep in onehot_rep:
    maximum=max(len(rep),maximum)
  embedded_docs=pad_sequences(onehot_rep,padding='pre',maxlen=maximum)
  return embedded_docs,maximum

def pre_process_text(sentence):
  sentence=lemmitization(sentence)
  test_embd,maximum=embedded_operation([sentence])
  return test_embd

def sentiment_analysis(sentence):
  test_embd=pre_process_text(sentence)
  model = tensorflow.keras.models.load_model('depression_analysis_model.keras')
  predict=model.predict(test_embd)
  predict=(predict>=0.5).astype("int")
  if predict[0][0]==1:
    print("The sentence sounds like a depressed sentence")
  else:
    print("This is a normal sentence")

