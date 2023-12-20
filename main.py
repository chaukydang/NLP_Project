from model import Models
from model import remove_punctuation, remove_stopwords, tokenization, lemmatizer
import pandas as pd
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
nltk.download('wordnet')
nltk.download("punkt")
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tkinter as tk
from tkinter import Entry, Text, Label, Button, Scrollbar
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
import numpy as np

import warnings
warnings.filterwarnings('ignore') # tắt những cảnh báo

df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('test.csv')
df_val = pd.read_csv('validation.csv')
df = pd.concat([df_train, df_test], axis = 0)

df['text'] = df['text'].apply(lambda x:remove_punctuation(x))
df['text'] = df['text'].apply(lambda x: tokenization(x))
df['text'] = df['text'].apply(lambda x:remove_stopwords(x))
df['text'] = df['text'].apply(lambda x:lemmatizer(x))
df['text'] = [item[0] for item in df['text']]

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
vect = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)

X_ml = vect.fit_transform(X)

X_train_ml, X_test_ml, y_train, y_test = train_test_split(X_ml,y, test_size=0.2, random_state=123)

# #Mô hình machine learning kết hợp adaboost
# Models.adaboost_Model(X_train_ml, X_test_ml, y_train, y_test)

# # #Mô hình Maximum Entropy
# # Models.maxent_Model(X_train, X_test, y_train, y_test)

# Mô hình Deep Learning CNN
Models.rnn_Model(df,df_train)
