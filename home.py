import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import pad_sequences, to_categorical
from keras.layers import Embedding, LSTM, Dense

def load_data():
    df = pd.read_csv(r"C:\Users\Yalavarthi Saadhika\Downloads\Netflix\netflix_reviews.csv")
    # Display first few rows of the dataframe for debugging
    return df


def create_and_train_model(df):
    texts = df["content"][0:200]
    token = Tokenizer()
    token.fit_on_texts(texts)

    lst1 = []
    lst2 = []
    for word in texts:
        sequences = token.texts_to_sequences([word])[0]
        for i in range(1, len(sequences)):
            lst1.append(sequences[:i])
            lst2.append(sequences[i])

    fv = pad_sequences(lst1)
    num_classes = len(token.word_index) + 1
    cv = to_categorical(lst2, num_classes=num_classes)

    model = Sequential()
    model.add(Embedding(input_dim=num_classes, output_dim=100, input_length=fv.shape[1]))
    model.add(LSTM(100))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(fv, cv, batch_size=30, epochs=50, validation_split=0.2)

    return model, token


df = load_data()


model, token = create_and_train_model(df)


st.title('Next Word Prediction APP')
st.write(":green[Enter the context for prediction:]")

input_text = st.text_input("Input text", ":red[Netflix]")
if st.button('Predict'):
    text = input_text
    st.write("Initial text:", text)
    for i in range(5):
        sequences = token.texts_to_sequences([text])
        padded_sequences = np.array(sequences)  # Convert sequences to numpy array
        v = np.argmax(model.predict(padded_sequences))
        if v in token.index_word:
            text += " " + token.index_word[v]
            st.write(text)
        else:
            st.write(f"Predicted index {v} is out of the vocabulary range.")
            break
