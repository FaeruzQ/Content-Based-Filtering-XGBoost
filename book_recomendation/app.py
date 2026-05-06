import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

from xgboost import XGBClassifier

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("Books.csv")

# =========================
# PREPROCESSING
# =========================

df.columns = df.columns.str.lower()

df = df[['title','author','genre','description']].dropna()

df.drop_duplicates(inplace=True)

placeholders = [
    'no description available',
    'unknown',
    'n/a'
]

mask = (
    df['description'].str.lower().str.strip().isin(placeholders)
)

df = df[~mask].reset_index(drop=True)

df['word_count'] = df['description'].apply(
    lambda x: len(str(x).split())
)

df = df[df['word_count'] >= 10]

# =========================
# FEATURE ENGINEERING
# =========================

df["combined_text"] = (
    df["title"].astype(str) + " " +
    df["author"].astype(str) + " " +
    df["genre"].astype(str) + " " +
    df["description"].astype(str)
)

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(
    df["combined_text"]
)

# =========================
# PSEUDO LABELING FUNCTION
# =========================

def create_labels_topk(sim_scores, k=0.2):

    n = len(sim_scores)

    top_k = int(n * k)

    idx = np.argsort(sim_scores)[-top_k:]

    labels = np.zeros(n)

    labels[idx] = 1

    return labels

# =========================
# TRAIN MODEL
# =========================

training_query = "Harry Potter"

query_vec = tfidf.transform([training_query])

sim_scores = cosine_similarity(
    query_vec,
    tfidf_matrix
).flatten()

X_features = hstack([
    tfidf_matrix,
    sim_scores.reshape(-1,1)
])

y = create_labels_topk(sim_scores, k=0.2)

model = XGBClassifier(
    random_state=42,
    n_estimators=100,
    eval_metric='logloss'
)

model.fit(X_features, y)

# =========================
# STREAMLIT UI
# =========================

st.title("Sistem Rekomendasi Buku")
st.write("Content-Based Filtering + XGBoost")

query = st.text_input(
    "Masukkan judul atau kata kunci buku"
)

if st.button("Cari Rekomendasi"):

    query_vec = tfidf.transform([query])

    sim_scores = cosine_similarity(
        query_vec,
        tfidf_matrix
    ).flatten()

    X_predict = hstack([
        tfidf_matrix,
        sim_scores.reshape(-1,1)
    ])

    relevance_probabilities = model.predict_proba(
        X_predict
    )[:,1]

    df_result = df.copy()

    df_result["similarity"] = sim_scores

    df_result["relevance_probability"] = (
        relevance_probabilities
    )

    top_books = df_result.sort_values(
        by="relevance_probability",
        ascending=False
    ).head(20)

    st.subheader("Rekomendasi Buku")

    st.dataframe(
        top_books[
            [
                "relevance",
                "title",
                "author",
                "genre"
            ]
        ]
    )
