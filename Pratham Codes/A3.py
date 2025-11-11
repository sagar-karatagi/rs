# Assignment 3: Collaborative Filtering Based Movie Recommendation System
# Author: Harsh Balkrishna Vahal

# --------------------------------------------------
# 🔹 Step 1: Import Libraries
# --------------------------------------------------
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

# --------------------------------------------------
# 🔹 Step 2: Load Dataset
# --------------------------------------------------
df = pd.read_csv("RS-A2_A3_Filtered_Ratings.csv")
print("✅ Dataset Loaded Successfully")
print(df.head())

# Drop unnecessary columns
df = df[['userId', 'movieId', 'rating']]

# --------------------------------------------------
# 🔹 Step 3: Create User–Item Matrix
# --------------------------------------------------
user_item_matrix = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
print("\n✅ User–Item Matrix Created")
print("Shape:", user_item_matrix.shape)

# --------------------------------------------------
# 🔹 Step 4: Compute User Similarity using Cosine Similarity
# --------------------------------------------------
similarity_matrix = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
print("\n✅ User Similarity Matrix Created")

# --------------------------------------------------
# 🔹 Step 5: Recommendation Function
# --------------------------------------------------
def recommend_movies(user_id, n=5):
    """
    Recommend top movies to a user based on similar users' preferences.
    """
    if user_id not in user_item_matrix.index:
        print("❌ User not found in dataset!")
        return

    # Find similar users
    sim_users = similarity_df[user_id].sort_values(ascending=False)[1:6]  # Top 5 similar users
    
    # Get their movie ratings
    similar_users_ratings = user_item_matrix.loc[sim_users.index]
    
    # Weighted average of ratings based on similarity scores
    weighted_ratings = similar_users_ratings.T.dot(sim_users) / sim_users.sum()
    
    # Remove movies the user has already rated
    user_rated_movies = user_item_matrix.loc[user_id]
    recommendations = weighted_ratings[user_rated_movies == 0].sort_values(ascending=False).head(n)
    
    print(f"\n🎬 Top {n} Recommended Movies for User {user_id}:\n")
    for movie_id, score in recommendations.items():
        print(f"👉 Movie ID: {movie_id}  (Predicted Rating: {score:.2f})")

# --------------------------------------------------
# 🔹 Step 6: Example Run
# --------------------------------------------------
example_user = user_item_matrix.index[0]  # first user
recommend_movies(example_user, n=5)
