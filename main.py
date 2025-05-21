import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_and_filter_data(movies_path, ratings_path, top_n_users=500):
    # Load data
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    print(ratings.columns.tolist())

    # Select top N users by userId ascending
    top_users = ratings['userId'].sort_values().unique()[:top_n_users]
    
    # Filter ratings for top users
    filtered_ratings = ratings[ratings['userId'].isin(top_users)]
    
    # Filter movies rated by these users
    filtered_movie_ids = filtered_ratings['movieId'].unique()
    filtered_movies = movies[movies['movieId'].isin(filtered_movie_ids)]
    
    return filtered_movies, filtered_ratings

def create_user_item_matrix(ratings_df):
    user_item = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return user_item

def compute_similarities(user_item_matrix):
    # User-based similarity matrix
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # Item-based similarity matrix
    item_sim = cosine_similarity(user_item_matrix.T)
    item_sim_df = pd.DataFrame(item_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    
    return user_sim_df, item_sim_df

def recommend_movies_user_based(user_id, user_item_matrix, user_similarity_df, top_n=5):
    if user_id not in user_similarity_df.index:
        print(f"User ID {user_id} not found.")
        return pd.Series()
    
    # Similarity scores for this user (excluding self)
    similar_users = user_similarity_df.loc[user_id].drop(user_id).sort_values(ascending=False)
    
    # Ratings of target user
    user_ratings = user_item_matrix.loc[user_id]
    
    # Weighted ratings sum from similar users
    weighted_sum = pd.Series(0, index=user_item_matrix.columns, dtype=float)
    sum_sim = 0
    
    for other_user, similarity in similar_users.items():
        if similarity <= 0:
            continue
        other_ratings = user_item_matrix.loc[other_user]
        weighted_sum += similarity * other_ratings
        sum_sim += similarity
    
    if sum_sim == 0:
        print("No similar users with positive similarity found.")
        return pd.Series()
    
    predicted_ratings = weighted_sum / sum_sim
    
    # Filter out movies already rated by user
    recommended = predicted_ratings[user_ratings == 0]
    
    return recommended.sort_values(ascending=False).head(top_n)

def recommend_similar_movies_item_based(movie_id, item_similarity_df, top_n=5):
    if movie_id not in item_similarity_df.index:
        print(f"Movie ID {movie_id} not found.")
        return pd.Series()
    
    similar_movies = item_similarity_df[movie_id].drop(movie_id).sort_values(ascending=False)
    return similar_movies.head(top_n)

def main():
    # Paths to your datasets
    ratings_path = "../../Dataset/ml-32m/ratings.csv"
    movies_path = "../../Dataset/ml-32m/movies.csv"
    
    print("Loading and filtering data...")
    movies, ratings = load_and_filter_data(movies_path, ratings_path)
    
    print(f"Filtered to {len(ratings['userId'].unique())} users and {len(movies)} movies.")
    
    print("Creating user-item matrix...")
    user_item_matrix = create_user_item_matrix(ratings)
    
    print("Computing similarity matrices...")
    user_sim_df, item_sim_df = compute_similarities(user_item_matrix)
    
    # Example user and movie for recommendations
    example_user = user_item_matrix.index[0]
    example_movie = user_item_matrix.columns[0]
    
    print(f"\nUser-based recommendations for user {example_user}:")
    recs_user = recommend_movies_user_based(example_user, user_item_matrix, user_sim_df, top_n=5)
    print(recs_user)
    
    print(f"\nItem-based recommendations similar to movie {example_movie}:")
    recs_item = recommend_similar_movies_item_based(example_movie, item_sim_df, top_n=5)
    print(recs_item)

if __name__ == "__main__":
    main()

