# Import Libraries
import pandas as pd
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
from google.colab import files


# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Read Dataset
df = pd.read_csv("/content/fake_job_postings.csv")
print(df.head())
print("\nFraudulent Value Counts:\n", df['fraudulent'].value_counts())

# Balance Dataset
df_0 = df[df['fraudulent'] == 0].sample(n=5000, random_state=42, replace=True)
df_1 = df[df['fraudulent'] == 1].sample(n=5000, random_state=42, replace=True)
df_balanced = pd.concat([df_0, df_1]).reset_index(drop=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare stopwords, lemmatizer, and stemmer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Clean missing and duplicate job descriptions
df_balanced.dropna(subset=['description'], inplace=True)
df_balanced.drop_duplicates(subset=['description'], inplace=True)

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


# Apply Cleaning
df_balanced['cleaned_job_desc'] = df_balanced['description'].apply(clean_text)
df_final = df_balanced[['cleaned_job_desc', 'fraudulent']]

print("\nSample cleaned data:\n", df_final.head())

# Train-Test Split
X = df_final['cleaned_job_desc']
y = df_final['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining Size: {len(X_train)} | Testing Size: {len(X_test)}")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF Train shape:", X_train_tfidf.shape)
print("TF-IDF Test shape:", X_test_tfidf.shape)

# Model Training
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print("\nModel Performance on Test Data:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save Model and Vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_job_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")
print("\n✅ Model and TF-IDF vectorizer saved successfully!")

# Function to Predict New Descriptions Safely
def predict_job_description(text, threshold=0.5):
    cleaned_text = clean_text(text)
    vectorized_text = tfidf.transform([cleaned_text])

    # If input has no recognized words
    if vectorized_text.nnz == 0:
        return "Unknown / Possibly gibberish"

    prob = model.predict_proba(vectorized_text)[0][1]  # probability of being 'fake'

    if prob >= threshold:
        return "Fake Job"
    else:
        return "Real Job"


# Example Usage
test_texts = [
    "We are looking for a software engineer with experience in Python and ML.",
    "asdfghjkl qwerty",
    "Marketing manager needed for international campaigns."
]

for text in test_texts:
    print(f"\nInput: {text}")
    print("Prediction:", predict_job_description(text))


# Download the model
files.download("model/fake_job_model.pkl")
files.download("model/tfidf_vectorizer.pkl")