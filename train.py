import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 🔥 STRONG MANUAL DATASET
data = {
    "text": [
        "This product is amazing and very good",
        "I love this item very much",
        "Highly recommend this product",
        "Very happy with the purchase",
        "Good quality and worth the money",

        "fake product scam do not buy",
        "worst product waste of money",
        "totally fake review spam",
        "this is fraud product very bad",
        "do not trust this fake item"
    ],
    "label": [0,0,0,0,0, 1,1,1,1,1]  # 0 = Genuine, 1 = Fake
}

df = pd.DataFrame(data)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train
model = LogisticRegression()
model.fit(X, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("🔥 Model ready!")