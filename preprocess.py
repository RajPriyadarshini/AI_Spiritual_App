import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("spiritual_data.csv")

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Save mapping
import pickle
pickle.dump(le, open("label_encoder.pkl", "wb"))

print(df.head())