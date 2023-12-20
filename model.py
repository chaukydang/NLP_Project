from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import nltk
from nltk.classify import MaxentClassifier
from nltk import FreqDist
from sklearn import svm
from sklearn.metrics import classification_report
import tensorflow
from tensorflow import keras
from keras import layers, models, optimizers, losses, metrics
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM
import string
string.punctuation
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import Entry, Text, Label, Button, Scrollbar
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def tokenization(text):
    tokens = re.split('W+',text)
    return tokens

def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

def lemmatizer(text):
  lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
  return lemm_text


class Models:
    def __init__(self):
        self.svm_Model = None
        self.maxent_Model = None
        self.cnn_Model = None

    def adaboost_Model(X_train, X_test, y_train, y_test):
        #Tạo mô hình TreeDecision Classification
        classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100)
        
        classifier.fit(X_train, y_train)
        
        y_pred_adaboost = classifier.predict(X_test)
        accuracy_Ada = accuracy_score(y_test, y_pred_adaboost)      
        print(f"Accuracy: {accuracy_Ada*100:.2f}%")
        print("Classification Report:\n", classification_report(y_test, y_pred_adaboost))


    def maxent_Model(X_train, y_train, X_test, y_test):
        # Xác định hàm trích xuất đặc trưng
        def extract_features(text):
            words = word_tokenize(text)
            return dict([(word, True) for word in words])

        # Chuẩn bị dữ liệu huấn luyện định dạng NLTK
        test_data = list(zip(X_test, y_test))
        train_set = [(extract_features(text), label) for text, label in zip(X_train, y_train)]
        # Train mô hình MaxEnt 
        maxent_model = MaxentClassifier.train(train_set, trace=0, algorithm='iis', max_iter = 20, min_lldelta=0.001)
        predictions = [maxent_model.classify(extract_features(text)) for text, _ in test_data]
        # Đánh giá mô hình trên tập xác thực
        validation_set = [(extract_features(text), label) for text, label in zip(X_test, y_test)]
        accuracy = nltk.classify.accuracy(maxent_model, validation_set)

        print(f"MaxEnt Model Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:\n", classification_report([label for _, label in test_data], predictions))
        return maxent_model

    def rnn_Model(data_train, data_validation):
        #tiền xử lý cho RNN:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data_train['text'])
        
        # Chuyển văn bản thành chuỗi
        train_sequences = tokenizer.texts_to_sequences(data_train['text'])
        val_sequences = tokenizer.texts_to_sequences(data_validation['text'])
        
        # Chuỗi pad có cùng độ dài
        max_sequence_length = max(len(seq) for seq in train_sequences)
        train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length)
        val_sequences = pad_sequences(val_sequences, maxlen=max_sequence_length)
        
        # RNN Model
        # Define the model architecture
        rnn_model = Sequential()
        rnn_model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length = max_sequence_length))
        rnn_model.add(LSTM(128))
        rnn_model.add(Dense(6, activation='softmax'))

        # Compile the model
        rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Train the model
        rnn_model.fit(train_sequences, data_train['label'], epochs=10, batch_size=32, validation_data=(val_sequences, data_validation['label']))

        # Chức năng dự đoán emotion
        def predict_emotion(input_text, output_text_widget):
            input_text = input_text.strip()
            if input_text:
                input_sequence = tokenizer.texts_to_sequences([input_text])
                padded_input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length)
                
                prediction = rnn_model.predict(padded_input_sequence)

                # Xác định emotion_labels
                emotion_labels = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

                # Điều chỉnh LabelEncode với các đối số phù hợp
                label_encoder = LabelEncoder()
                label_encoder.fit(list(emotion_labels.keys()))
                                
                # Chuyển đổi ngược predicted label
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
                emotion_labels = emotion_labels[label_encoder.inverse_transform([np.argmax(prediction[0])])[0]]

                # Cập nhật output text widget
                output_text_widget.config(state="normal")
                output_text_widget.delete(1.0, "end")
                output_text_widget.insert("end", f"Predicted Emotion: {predicted_label}: {emotion_labels}")
                output_text_widget.config(state="disabled")


        # Tạo GUI
        root = tk.Tk()
        root.title("Emotion Prediction")

        # Định nghĩa các phần tử GUI
        input_label = Label(root, text="Enter Text:")
        input_entry = Entry(root, width=50)
        output_label = Label(root, text="Prediction:")
        output_text = Text(root, height=5, width=50, state="disabled")
        scrollbar = Scrollbar(root, command=output_text.yview)
        output_text.config(yscrollcommand=scrollbar.set)
        predict_button = Button(root, text="Predict Emotion", command=lambda: predict_emotion(input_entry.get(), output_text))

        # Đặt các phần tử GUI trên cửa sổ
        input_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        input_entry.grid(row=0, column=1, padx=10, pady=10)
        output_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        output_text.grid(row=1, column=1, padx=10, pady=10)
        scrollbar.grid(row=1, column=2, pady=10, sticky="ns")
        predict_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Chạy vòng lặp chính của GUI
        root.mainloop()


