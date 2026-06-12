#  imports 
import streamlit as st 
import joblib 
import pandas as pd 
import time 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import os 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 
 
# Page config 
st.set_page_config(page_title="Fake Job Detector", page_icon="🧑‍💻", 
layout="wide") 
 
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.upay.org.in/wp-content/uploads/2020/04/Seccond-Career-Header-Background.png");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}

.stContainer > div {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

body, h1, h2, h3, p, span {
    color: #000000;
    font-family: 'Arial', sans-serif;
}

.stButton>button {
    background: linear-gradient(90deg, #f0f0f0, #d9d9d9);
    color: black;
    font-weight: bold;
    padding: 12px 25px;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: #007bff;
    color: white;
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}

textarea {
    background-color: rgba(255,255,255,0.9) !important;
    color: #000000 !important;
    border-radius: 15px !important;
    border: 1px solid rgba(0,0,0,0.2) !important;
    padding: 15px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    resize: vertical !important;
    min-height: 150px;
}

.prediction-bar {
    border-radius: 12px;
    padding: 8px 0;
    text-align: center;
    font-weight: bold;
    color: white;
}
</style>
""", unsafe_allow_html=True)

 
# Load NLTK resources  
nltk.download("stopwords", quiet=True) 
nltk.download("wordnet", quiet=True) 
stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer() 
 
# Feedback CSV  
FEEDBACK_CSV = "data/feedback_data.csv" 
if not os.path.exists("data"): 
    os.makedirs("data") 
if not os.path.exists(FEEDBACK_CSV): 
    pd.DataFrame(columns=["job_description","predicted","correct"]).to_csv(FEEDBACK_CSV,index=False) 
 
# Load model + TF-IDF  
@st.cache_resource 
def load_resources(): 
    model = joblib.load("model/fake_job_model (5).pkl") 
    tfidf = joblib.load("model/tfidf_vectorizer (3).pkl") 
    return model, tfidf 
 
# Load dataset  
@st.cache_data 
def load_data(): 
    return pd.read_csv("data/fake_job_postings.csv") 
 
# Text cleaning  
def clean_text(text): 
    text = str(text).lower() 
    text = re.sub(r"http\S+|www\S+|https\S+", "", text) 
    text = re.sub(r"<.*?>", "", text) 
    text = re.sub(r"\s+", " ", text).strip() 
    tokens = re.findall(r"[a-zA-Z0-9#+]+", text) 
    tokens = [word for word in tokens if word not in stop_words and len(word) > 
1] 
    tokens = [lemmatizer.lemmatize(word) for word in tokens] 
    return " ".join(tokens) 
 
# Prediction function  
def predict_job_description(text, model, tfidf, threshold=0.5): 
    cleaned_text = clean_text(text) 
    vectorized_text = tfidf.transform([cleaned_text]) 
    if vectorized_text.nnz == 0: 
        return "⚠️ Unknown / Possibly gibberish", None, None 
    prob_fake = model.predict_proba(vectorized_text)[0][1] 
    prediction = 1 if prob_fake >= threshold else 0 
    confidence = round(max(prob_fake, 1 - prob_fake) * 100, 2) 
    result = "Fake Job" if prediction == 1 else "Real Job" 
    bar_color = "#ff4b4b" if prediction == 1 else "#4caf50" 
    return result, confidence, bar_color 
 
# Retrain model with feedback  
def retrain_model(): 
    df = pd.read_csv("data/fake_job_postings.csv") 
    df_feedback = pd.read_csv(FEEDBACK_CSV) 
    if not df_feedback.empty: 
        df_feedback_renamed = df_feedback.rename(columns={"job_description":"description","correct":"fraudulent"}) 
        df_combined = pd.concat([df[['description','fraudulent']], 
df_feedback_renamed]) 
        df_combined.dropna(subset=['description'], inplace=True) 
        df_combined.drop_duplicates(subset=['description'], inplace=True) 
        df_combined['cleaned_job_desc'] = df_combined['description'].apply(clean_text) 
 
        X = df_combined['cleaned_job_desc'] 
        y = df_combined['fraudulent'] 
 
        tfidf_new = TfidfVectorizer(max_features=5000, ngram_range=(1,2), 
stop_words='english') 
        X_tfidf = tfidf_new.fit_transform(X) 
        model_new = LogisticRegression(max_iter=1000, class_weight="balanced", 
random_state=42) 
        model_new.fit(X_tfidf, y) 
 
        os.makedirs("model", exist_ok=True) 
        joblib.dump(model_new,"model/fake_job_model (5).pkl") 
        joblib.dump(tfidf_new,"model/tfidf_vectorizer (3).pkl") 
        return model_new, tfidf_new 
    return None, None 
 
# App Title  
st.markdown("<div style='text-align: center;'><h1>🧑‍💻 Fake Job Detector</h1><p>Detect whether a job posting is real or fake using ML & TFIDF!</p></div>", unsafe_allow_html=True) 
 
# Load resources  
with st.spinner("Loading model and TF-IDF vectorizer..."): 
    model, tfidf = load_resources() 
st.success("✅ Model loaded!") 
 
# Load dataset  
with st.spinner("Loading dataset..."): 
    df = load_data() 
st.success("✅ Dataset loaded!") 
 
# Dataset preview  
st.subheader("📊 Sample Dataset Preview") 
st.dataframe(df.head(), height=200) 
 
# Job description input  
st.subheader("📝 Check a Single Job Description") 
job_text = st.text_area("Paste job description here", height=150) 
 
#  Session state  
if "predicted" not in st.session_state: st.session_state.predicted = False 
if "feedback_given" not in st.session_state: st.session_state.feedback_given = False 
 
# Predict button  
if st.button("Predict") and job_text.strip() != "": 
    result, confidence, bar_color = predict_job_description(job_text, model, 
tfidf) 
    st.session_state.predicted = True 
    st.session_state.pred_result = result 
    st.session_state.confidence = confidence 
    st.session_state.bar_color = bar_color 
 
if st.session_state.predicted: 
    if st.session_state.pred_result.startswith("⚠️"): 
        st.warning(st.session_state.pred_result) 
    else: 
        st.markdown(f"<div class='prediction-bar' style='width:{st.session_state.confidence}%; background:{st.session_state.bar_color};'>{st.session_state.pred_result} – Confidence: {st.session_state.confidence}%</div>", unsafe_allow_html=True) 
 
    # Feedback buttons  
    if not st.session_state.feedback_given: 
        col1, col2 = st.columns(2) 
        with col1: 
            if st.button("✅ Correct"): 
                st.session_state.feedback_given = True 
                feedback_df = pd.read_csv(FEEDBACK_CSV) 
                feedback_df = pd.concat([feedback_df, 
pd.DataFrame([{"job_description": job_text,"predicted": 
st.session_state.pred_result,"correct": 1}])], ignore_index=True) 
                feedback_df.to_csv(FEEDBACK_CSV,index=False) 
                st.success("Feedback recorded: Correct!") 
        with col2: 
            if st.button("❌ Wrong"): 
                st.session_state.feedback_given = True 
                feedback_df = pd.read_csv(FEEDBACK_CSV) 
                feedback_df = pd.concat([feedback_df, 
pd.DataFrame([{"job_description": job_text,"predicted": 
st.session_state.pred_result,"correct": 0}])], ignore_index=True) 
                feedback_df.to_csv(FEEDBACK_CSV,index=False) 
                st.warning("Feedback recorded: Wrong!") 
 
    # Retrain button  
    if st.session_state.feedback_given: 
        if st.button("🔄 Retrain / Reload Model"): 
            with st.spinner("Retraining model with feedback..."): 
                model, tfidf = retrain_model() 
                st.session_state.predicted = False 
                st.session_state.feedback_given = False 
                st.success("✅ Model retrained successfully! Ready for next predictions.")