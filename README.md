# Collaborative Filtering Recommender System

This project implements **User-based** and **Item-based Collaborative Filtering** using **Cosine Similarity** on a movie rating dataset.

---

## Dataset

- `movies.csv`: Contains movie metadata with columns:
  - `movieId`
  - `title`
  - `genres`
  
- `ratings.csv`: Contains user ratings with columns:
  - `userId`
  - `movieId`
  - `rating`
  - `timestamp`

---

## Approach

1. **Filter top 500 users by `userId` (ascending)** to reduce dataset size and computational cost.
2. Select movies rated by these users.
3. Create a user-item rating matrix.
4. Calculate cosine similarity matrices:
   - User-User similarity for user-based collaborative filtering.
   - Item-Item similarity for item-based collaborative filtering.
5. Generate recommendations based on these similarity matrices.

---

## How to Run

1. Ensure you have Python 3.x installed.
2. Install required libraries:

```bash
pip install pandas scikit-learn
