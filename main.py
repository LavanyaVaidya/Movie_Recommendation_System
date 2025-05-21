import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
ratings = pd.read_csv("../../Dataset/ml-32m/ratings.csv")   # userId, movieId, rating
movies = pd.read_csv("../../Dataset/ml-32m/movies.csv")     # movieId, title

# Create user-item matrix
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# --- USER-BASED COLLABORATIVE FILTERING ---
def user_based_recommendation(user_id, top_n=5):
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Get similar users
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:]

    weighted_ratings = pd.Series(dtype=np.float64)

    for sim_user, score in similar_users.items():
        sim_user_ratings = user_item_matrix.loc[sim_user]
        weighted_ratings = weighted_ratings.add(sim_user_ratings * score, fill_value=0)

    # Normalize by similarity sum
    weighted_ratings /= similar_users.sum()

    # Remove already rated movies
    user_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = weighted_ratings.drop(user_rated).sort_values(ascending=False).head(top_n)

    return movies[movies.movieId.isin(recommendations.index)][['movieId', 'title']]

# --- ITEM-BASED COLLABORATIVE FILTERING ---
def item_based_recommendation(movie_title, top_n=5):
    # Transpose for item similarity
    item_user_matrix = user_item_matrix.T

    # Compute cosine similarity between items
    item_sim = cosine_similarity(item_user_matrix)
    item_sim_df = pd.DataFrame(item_sim, index=item_user_matrix.index, columns=item_user_matrix.index)

    # Find movieId for the title
    target_movie_id = movies[movies.title == movie_title].movieId.values[0]

    # Get similar items
    similar_items = item_sim_df[target_movie_id].sort_values(ascending=False)[1:top_n+1]
    similar_movie_ids = similar_items.index

    return movies[movies.movieId.isin(similar_movie_ids)][['movieId', 'title']]

# Example usage
print("User-based recommendations for user 10:")
print(user_based_recommendation(user_id=10))

print("\nItem-based recommendations for 'Toy Story (1995)':")
print(item_based_recommendation(movie_title='Toy Story (1995)'))
