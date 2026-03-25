import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Load data
df = pd.read_csv("spiritual_data.csv")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text']).toarray()

le = LabelEncoder()
y = le.fit_transform(df['label'])

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Model
class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Model(X.shape[1])

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(100):
    outputs = model(X)
    loss = loss_fn(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "torch_model.pth")