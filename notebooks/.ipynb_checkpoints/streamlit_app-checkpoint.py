import streamlit as st
import os
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import numpy as np
import pandas as pd
#openai_key = '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'

st.title('TuneIn Therapies')

user_input = st.text_input("Tell me how you are feeling today.", "I am feeling sad, as if life is not fulfilling.")

save_directory = "/saved_models"

loaded_tokenizer = DistilBertTokenizer.from_pretrained(save_directory)
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

predict_input = loaded_tokenizer.encode(user_input,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")

output = loaded_model(predict_input)[0]

def score_each_class(output):
    #softmax will give the output as probabilities
    softmax = tf.nn.softmax(output)
    
    #convert to numpy
    scores = softmax.numpy()[0]
    
    #multiply by 10 to get scores out of 10
    scores = scores*10
    
    return np.round(scores, 2)

emotion_vals = score_each_class(output)

emotion_dct = {'Fear': emotion_vals[0], 'Sadness': emotion_vals[1], 'Worry': emotion_vals[2]}

#get anxiety level from fear and worry
fear_score = emotion_dct['Fear']
worry_score = emotion_dct['Worry']

min_score = min(fear_score, worry_score)
max_score = max(fear_score, worry_score)

scaled_fear_score = 10 * ((fear_score - min_score) / (max_score - min_score))
scaled_worry_score = 10 * ((worry_score - min_score) / (max_score - min_score))
#anxiety is the weighted average
anxiety_score = (scaled_fear_score + scaled_worry_score)

#st.write(pd.DataFrame(emotion_dct, index=['Scores']))
st.write('Your sadness score is:', emotion_dct['Sadness'])
st.write('Your anxiety score is:', anxiety_score)





