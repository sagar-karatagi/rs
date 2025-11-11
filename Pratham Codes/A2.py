# Assignment 2: Content-Based Movie Recommendation System
# Author: Harsh Balkrishna Vahal

# ------------------------------------------------
# 🔹 Step 1: Import Libraries
# ------------------------------------------------
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------
# 🔹 Step 2: Load Dataset
# ------------------------------------------------
df = pd.read_csv("RS-A2_A3_movie.csv")
print("✅ Dataset Loaded Successfully")
print(df.head())

# ------------------------------------------------
# 🔹 Step 3: Data Preprocessing
# ------------------------------------------------
# Replace '|' with space for better tokenization
df['genres'] = df['genres'].str.replace('|', ' ')
df = df.dropna(subset=['genres'])

print("\n✅ Cleaned Data (Genres processed):")
print(df.head())

# ------------------------------------------------
# 🔹 Step 4: TF-IDF Vectorization on Genres
# ------------------------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])

print("\n✅ TF-IDF Matrix Shape:", tfidf_matrix.shape)

# ------------------------------------------------
# 🔹 Step 5: Compute Cosine Similarity
# ------------------------------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("\n✅ Cosine Similarity Computed")

# ------------------------------------------------
# 🔹 Step 6: Recommendation Function
# ------------------------------------------------
def recommend_movies(title, n=5):
    """
    Recommend top-n movies similar to the given movie based on genre similarity.
    """
    if title not in df['title'].values:
        print("❌ Movie not found! Try an exact title from the dataset.")
        return
    
    idx = df.index[df['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # skip itself
    
    print(f"\n🎬 Top {n} Recommendations for '{title}':\n")
    for i, score in sim_scores:
        print(f"👉 {df.iloc[i]['title']}  ({df.iloc[i]['genres']})")

# ------------------------------------------------
# 🔹 Step 7: Example Run
# ------------------------------------------------
recommend_movies("Toy Story (1995)", n=5)
