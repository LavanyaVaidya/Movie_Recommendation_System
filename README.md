# 🎬 Movie Recommendation System

This project implements a **movie recommendation system** using:
- **User-Based Collaborative Filtering**
- **Item-Based Collaborative Filtering**
- **Cosine Similarity**

It is built using **Python**, **pandas**, and **scikit-learn**, with optimizations to handle large datasets.

---

## 📁 Dataset

We use the [MovieLens dataset](https://grouplens.org/datasets/movielens/) which contains:

- `movies.csv`: Information about movies (movieId, title, genres)
- `ratings.csv`: User ratings of movies (userId, movieId, rating, timestamp)

> ⚠️ If working with the full dataset (27M+ ratings), memory optimizations are included to prevent RAM issues.

---

## 💡 How it Works

### ✅ User-Based Collaborative Filtering
- Compares users to find similar taste profiles using **cosine similarity**.
- Predicts ratings for a user based on ratings from similar users.
- Output: Movie recommendations personalized to the target user.

### ✅ Item-Based Collaborative Filtering
- Compares movies based on how similar users rated them.
- Predicts ratings by looking at how similar movies were rated by a user.
- Output: Movies similar to those a user already liked.

---

## ⚙️ Technologies Used

- Python 3.x
- pandas
- numpy
- scikit-learn (for cosine similarity)
- scipy (for sparse matrices)

---

## 🧪 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/movie-recommender
   cd movie-recommender
