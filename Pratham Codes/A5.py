# Assignment 5: Matrix Factorization-based Recommendation System
# Author: Harsh Balkrishna Vahal

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# Step 1: Load Dataset
df = pd.read_csv("RS-A5_amazon_products_sales_data_cleaned.csv")
print("✅ Dataset Loaded Successfully")
df.head()   

# Step 2: Select Relevant Columns
# We'll use product_category (user), product_title (item), and product_rating (rating)
df = df[['product_title', 'product_category', 'product_rating']]
df = df.dropna(subset=['product_rating'])

print("✅ Selected Columns: product_title, product_category, product_rating")
print("Shape:", df.shape)
df.head()


# Step 3: Create Pivot Table for Matrix Factorization
pivot = df.pivot_table(index='product_category', columns='product_title', values='product_rating', fill_value=0)

print("✅ User–Item Matrix Created")
print("Shape:", pivot.shape)
pivot.head()


# Step 4: Apply SVD to Decompose the User–Item Matrix
svd = TruncatedSVD(n_components=10, random_state=42)
latent_matrix = svd.fit_transform(pivot)

print("✅ SVD Applied Successfully")
print("Latent Feature Matrix Shape:", latent_matrix.shape)


# Step 5: Reconstruct Predicted Ratings
pred_matrix = np.dot(latent_matrix, svd.components_)
pred_df = pd.DataFrame(pred_matrix, index=pivot.index, columns=pivot.columns)

# Step 6: Evaluate with RMSE
rmse = np.sqrt(mean_squared_error(pivot.values.flatten(), pred_matrix.flatten()))
print("📊 Model RMSE:", round(rmse, 3))


# Step 7: Static Recommendation Function (Non-interactive)
def recommend_products(category_name, n=5):
    if category_name not in pred_df.index:
        print("❌ Category not found!")
        print("Available categories:", list(pred_df.index))
        return
    
    sorted_products = pred_df.loc[category_name].sort_values(ascending=False).head(n)
    print(f"\n🔹 Top {n} Product Recommendations for '{category_name}' customers:")
    for i, (product, score) in enumerate(sorted_products.items(), 1):
        print(f"{i}. {product}  (Predicted Rating: {score:.2f})")

# Example Run (Change category name as per your dataset)
recommend_products("Phones", n=5)



# Step 8: Interactive Recommendation System
# Allows user to input category name

def recommend_products(category_name, n=5):
    """
    Recommend top-n products for a selected product category.
    """
    if category_name not in pred_df.index:
        print(f"❌ Category '{category_name}' not found! Try one of these:")
        print(list(pred_df.index))
        return
    
    sorted_products = pred_df.loc[category_name].sort_values(ascending=False).head(n)
    print(f"\n🔹 Top {n} Product Recommendations for '{category_name}' customers:")
    for i, (product, score) in enumerate(sorted_products.items(), 1):
        print(f"{i}. {product}  (Predicted Rating: {score:.2f})")

# Interactive Input Loop
while True:
    user_input = input("\nEnter a product category to see recommendations (or type 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        print("👋 Exiting Recommendation System. Thank you!")
        break
    recommend_products(user_input, n=5)

