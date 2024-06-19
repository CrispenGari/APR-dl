
import os
import nltk


import numpy as np
import tensorflow as tf
import json

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize

nltk.download("punkt")

with open(os.path.join(os.getcwd(), "model/static/vocab.json"), 'r') as ref:
    VOCAB = json.load(ref)
    
print(" ✅ LOADING 'upvotes arrays' MODEL!\n") 
train_up_votes = np.load(os.path.join(os.getcwd(), "model/static/upvotes.npy"))
print(" ✅ LOADING 'upvotes arrays' MODEL DONE!\n")
    
# variables
max_words = 100
vocab_size = 9173
scaler = MinMaxScaler()
scaler.fit(train_up_votes)
MODEL_PATH_NAME = os.path.join(os.getcwd(), "model/static/apr-model.h5")

def text_to_sequence(sent):
    words = word_tokenize(sent.lower())
    sequences = []
    for word in words:
        try:
            sequences.append(VOCAB[word])
        except Exception:
            sequences.append(0)
    return sequences

class APR(keras.Model):
    def __init__(self):
        super(APR, self).__init__()
        # layers for bidirectional
        forward_layer = keras.layers.GRU(
        128, return_sequences=True, dropout=.5,
        name="gru_forward_layer"
        )
        backward_layer = keras.layers.LSTM(
        128, return_sequences=True, dropout=.5,
        go_backwards=True, name="lstm_backward_layer"
        )
        self.embedding = keras.layers.Embedding(
            vocab_size, 100, 
            input_length=max_words,
            # weights=[embedding_matrix], 
            trainable=True,
            name = "embedding_layer"
        )
        self.bidirectional = keras.layers.Bidirectional(
            forward_layer,
            backward_layer = backward_layer,
            name= "bidirectional_layer"
        )
        self.gru_layer = keras.layers.GRU(
                512, return_sequences=True,
                dropout=.5,
                name= "gru_layer"
        )
        self.lstm_layer = keras.layers.LSTM(
                512, return_sequences=True,
                dropout=.5,
                name="lstm_layer"
        )
        self.fc_1 = keras.layers.Dense(512, activation="relu", name="upvote_fc1")
        self.pooling_layer = keras.layers.GlobalAveragePooling1D(
            name="average_pooling_layer"
        )
        self.concatenate_layer = keras.layers.Concatenate(name="concatenate_layer_layer")

        self.dense_1 = keras.layers.Dense(64, activation='relu', name="dense_1")
        self.dropout_1 = keras.layers.Dropout(rate= .5, name="dropout_layer_0")
        self.dense_2 = keras.layers.Dense(512, activation='relu', name="dense_2")
        self.dropout_2 =  keras.layers.Dropout(rate= .5, name="dropout_layer_1")
        self.dense_3 = keras.layers.Dense(128, activation='relu', name="dense_3")
        self.dropout_3 = keras.layers.Dropout(rate= .5, name="dropout_layer_2")
        self.rating_output = keras.layers.Dense(5, activation='softmax', name="rating_output")
        self.recommend_output = keras.layers.Dense(1, activation='sigmoid', name="recommend_output")
            
    def call(self, inputs):
        text, upvote = inputs
        # Leaning the text features
        x_1 = self.embedding(text)
        x_1 = self.bidirectional(x_1)
        x_1 = self.gru_layer(x_1)
        x_1 = self.lstm_layer(x_1)
        x_1 = self.pooling_layer(x_1)

        # Learning the upvotes
        x_2 = self.fc_1(upvote)

        # concatenation
        x = self.concatenate_layer([x_1, x_2])

        # leaning combinned features
        x = self.dense_1(self.dropout_1(x))
        x = self.dense_2(self.dropout_2(x))
        x = self.dense_3(self.dropout_3(x))

        # outputs
        rating = self.rating_output(x)
        recommend = self.recommend_output(x)
        return rating, recommend
    
print(" ✅ LOADING TENSORFLOW APR MODEL!\n") 
apr_model = APR()
apr_model.build([(None, 100), (None, 1)])
apr_model.load_weights(MODEL_PATH_NAME)
print(" ✅ LOADING TENSORFLOW APR MODEL DONE!\n")

def apr_predictor(text_review: str, text_review_upvote:int, model):
    
    recomment_classes =["NOT RECOMMENDED", "RECOMMENDED"]
    recomment_emoji = ["👎", "👍"]
    tokens = text_to_sequence(text_review)
    padded_tokens = keras.preprocessing.sequence.pad_sequences([tokens],
                                    maxlen=max_words,
                                    padding="post", 
                                    truncating="post"
                                    )
    text_review_upvote = scaler.transform(np.array([text_review_upvote]).reshape(-1, 1))
    inputs = [padded_tokens, text_review_upvote]
    rating_pred, recommend_pred = model.predict(inputs)

    rating_pred = np.squeeze(rating_pred)
    rating_label = np.argmax(rating_pred)
    recomment_pred = tf.squeeze(recommend_pred).numpy()
    recomment_label = 1 if recomment_pred >=0.5 else 0
    probability = float(round(recomment_pred, 3)) if recomment_pred >= 0.5 else float(round(1 - recomment_pred, 3))

    pred_obj ={
        "recommend": {
            "label": recomment_label,
            "probability": probability,
            "class_": recomment_classes[recomment_label],
            "emoji": recomment_emoji[recomment_label]
        },
        "rating":{
            "rating": rating_label+1,
            "stars": "⭐" * (rating_label +1),
            "probability": float(round(rating_pred[rating_label], 3))
        }
    }
    return pred_obj