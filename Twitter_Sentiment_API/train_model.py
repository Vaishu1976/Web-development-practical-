import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None,
    low_memory=False
)

# FORCE target to numeric
df[0] = pd.to_numeric(df[0], errors="coerce")

df.columns = ["target", "id", "date", "flag", "user", "text"]

# Drop invalid rows
df = df.dropna(subset=["target"])

# Keep only 0 and 4
df = df[df["target"].isin([0, 4])]

# Map labels
df["target"] = df["target"].map({0: "Negative", 4: "Positive"})

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].astype(str).apply(clean_text)

print("Dataset size after cleaning:", df.shape)

# SAFE sampling
df = df.sample(n=min(50000, len(df)), random_state=42)

X = df["text"]
y = df["target"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model & Vectorizer saved")
