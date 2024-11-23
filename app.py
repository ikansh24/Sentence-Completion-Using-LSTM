import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the saved model and tokenizer
model = tf.keras.models.load_model('sentence_completion_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define max sequence length
max_sequence_len = 35  # Adjust based on your training data

def predict_top_five_words(model, tokenizer, seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    top_five_indexes = np.argsort(predicted[0])[::-1][:5]
    top_five_words = []
    for index in top_five_indexes:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                top_five_words.append(word)
                break
    return top_five_words

# Streamlit UI components
st.title("Sentence Auto Completion App")
st.write("Enter a sentence, and the model will predict the next top 5 words.")

user_input = st.text_input("Enter a sentence:")

if user_input:
    top_five_words = predict_top_five_words(model, tokenizer, user_input)
    st.write(f"Top 5 predicted words after '{user_input}':")
    st.write(top_five_words)
