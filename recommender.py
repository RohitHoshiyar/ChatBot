import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load data
users_df = pd.read_csv("data/Updated_Users_Dataset_with_Demographics.csv")
products_df = pd.read_csv("data/products_large.csv")
ratings_df = pd.read_csv("data/ratings_large.csv")

# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create FAISS index
descriptions = products_df['description'].fillna("").tolist()
embeddings = model.encode(descriptions, show_progress_bar=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Preprocess user data
gender_encoded = pd.get_dummies(users_df['gender'], prefix='gender')
location_map = {loc: idx for idx, loc in enumerate(users_df['location'].unique())}
users_df['location_encoded'] = users_df['location'].map(location_map)
scaler = MinMaxScaler()
users_df['age_scaled'] = scaler.fit_transform(users_df[['age']])
interests_split = users_df['interests'].str.get_dummies(sep=',')

user_features = pd.concat([
    users_df[['user_id', 'location_encoded', 'age_scaled']],
    gender_encoded,
    interests_split
], axis=1)

def recommend_for_user(user_id, query, top_n=5):
    if user_id not in users_df['user_id'].values:
        return semantic_search(query)

    user_vec = user_features[user_features['user_id'] == user_id].drop(columns=['user_id']).values
    all_vecs = user_features.drop(columns=['user_id']).values

    sim = cosine_similarity(user_vec, all_vecs)[0]
    user_features['similarity'] = sim
    similar_users = user_features.sort_values(by='similarity', ascending=False).head(10)['user_id'].values
    sim_ratings = ratings_df[ratings_df['user_id'].isin(similar_users)]

    merged = sim_ratings.merge(user_features[['user_id', 'similarity']], on='user_id')
    merged['weighted_rating'] = merged['rating'] * merged['similarity']
    agg = merged.groupby('product_id')['weighted_rating'].sum().reset_index().sort_values(by='weighted_rating', ascending=False)

    results = products_df[products_df['product_id'].isin(agg.head(top_n)['product_id'])]
    return results[['name', 'description', 'price']].to_dict(orient="records")

def semantic_search(query, top_k=5):
    query_vec = model.encode([query])[0].astype("float32")
    _, indices = index.search(np.array([query_vec]), top_k)
    results = products_df.iloc[indices[0]]
    return results[['name', 'description', 'price']].to_dict(orient="records")
