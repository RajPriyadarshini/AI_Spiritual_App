import streamlit as st
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model("tf_spiritual_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.title("🌿 Spiritual AI Classifier")

text = st.text_input("Enter your spiritual thought:")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq)

    pred = model.predict(padded)
    label = le.inverse_transform([pred.argmax()])

    st.success(f"Category: {label[0]}")