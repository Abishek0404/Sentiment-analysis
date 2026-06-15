import pickle
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import preprocess

print("Loading dataset...")
dataset = load_dataset("stanfordnlp/imdb", cache_dir="C:/Users/acer/.cache/huggingface/hub")
train_data = dataset['train'].shuffle(seed=42).select(range(2000))
train_texts = [preprocess(x) for x in train_data['text']]
train_labels = train_data['label']

print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_texts)

print("Training...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# Save model and vectorizer to disk
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Done! Model saved.")