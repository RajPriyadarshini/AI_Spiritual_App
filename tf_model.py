import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("spiritual_data.csv")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['label'])
pickle.dump(le, open("label_encoder.pkl", "wb"))

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['text'])

X = tokenizer.texts_to_sequences(df['text'])
X = tf.keras.preprocessing.sequence.pad_sequences(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20)

model.save("tf_spiritual_model.h5")

# Save tokenizer
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))