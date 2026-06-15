🎬 AI-Based Sentiment Analysis System

A Machine Learning project that analyzes movie reviews and predicts whether they are Positive or Negative using NLP techniques.


📌 Project Overview

This system uses a classic NLP classification pipeline:

Raw Text → Preprocessing → TF-IDF Vectorization → Logistic Regression → Prediction


Dataset: IMDB Movie Reviews (25,000 train / 25,000 test)
Model: Logistic Regression
Accuracy: ~84%
UI: Streamlit web app



📁 Project Structure

sentiment-analysis/
├── preprocess.py      # Text cleaning pipeline
├── train.py           # Model training and saving
├── app.py             # Streamlit web app
├── model.pkl          # Saved model (generated after training)
└── vectorizer.pkl     # Saved vectorizer (generated after training)


⚙️ Prerequisites


Python 3.11 — download from python.org
VS Code — download from code.visualstudio.com



🚀 How to Run

Step 1 — Place all project files in one folder

Make sure preprocess.py, train.py, and app.py are all in the same folder.

Step 2 — Install dependencies

Open terminal in the project folder and run:

bashpip install scikit-learn nltk datasets streamlit

Step 3 — Train the model

bashpython train.py

Wait for the message: "Done! Model saved."

This will create model.pkl and vectorizer.pkl in your folder. You only need to do this once.

Step 4 — Run the app

bashstreamlit run app.py

Step 5 — Use it

Your browser will open at http://localhost:8501. Type any movie review and click Analyze.


🧠 How It Works

1. Preprocessing (preprocess.py)


Converts text to lowercase
Removes HTML tags (e.g. <br />)
Removes punctuation and numbers
Removes stopwords (e.g. "the", "is", "and")
Lemmatizes words to their root form (e.g. "loved" → "love")


2. TF-IDF Vectorization

Converts cleaned text into numerical features. TF-IDF measures how important a word is in a review relative to all reviews — common words get lower scores, unique/meaningful words get higher scores.

3. Logistic Regression

Learns which words are associated with positive vs negative reviews and makes predictions based on their weighted combination.


⚠️ Known Limitations


Sarcasm detection is unreliable — the model looks at individual words, not context. "Groundbreaking" will lean positive even in a sarcastic sentence.
Trained on only 2000 samples for speed — accuracy can be improved by training on the full 25,000 reviews on a more powerful machine.
Binary classification only (Positive/Negative) — Neutral class not yet implemented.



🔮 Future Improvements


Use DistilBERT (transformer model) for context-aware predictions and better sarcasm handling (~95% accuracy)
Add Neutral class for low-confidence predictions
Train on full dataset using Google Colab (free GPU)
Add prediction history table in the UI



🛠️ Tech Stack

ToolPurposePython 3.11Core languageNLTKText preprocessingScikit-learnTF-IDF + Logistic RegressionHuggingFace DatasetsIMDB datasetStreamlitWeb UI
