# Recommendations with IBM Watson Studio

## Udacity – Unsupervised Learning Nanodegree | Recommendation Systems Project

---

## Project Overview

This project analyzes user interactions with articles on the IBM Watson Studio platform and builds a multi-method recommendation system to personalize the user experience.

---

## Files

| File | Description |
|------|-------------|
| `Recommendations_with_IBM.ipynb` | Main project notebook |
| `project_tests.py` | Test helpers for rubric validation |
| `data/user-item-interactions.csv` | User-article interaction log |
| `data/articles_community.csv` | Article metadata |

---

## Recommendation Methods

### I. Exploratory Data Analysis
- Dataset statistics: users, articles, interactions
- Distribution of user activity
- Most interacted articles

### II. Rank-Based Recommendations
- Most popular articles by interaction count
- Fallback for new users (cold-start problem)

### III. User-User Collaborative Filtering
- User-item binary interaction matrix
- Dot-product similarity between users
- Ranked recommendations from similar users

### IV. Content-Based Recommendations
- TF-IDF vectorization of article text
- KMeans clustering for article groups
- Cosine similarity for article-article recommendations

### V. Matrix Factorization (SVD)
- Singular Value Decomposition of the user-item matrix
- Latent feature analysis and selection
- Article-article recommendations via SVD embeddings

---

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib scipy jupyter
```

Place the data files in a `data/` folder:
```
data/
  user-item-interactions.csv
  articles_community.csv
```

Then launch:
```bash
jupyter notebook Recommendations_with_IBM.ipynb
```

---

## Rubric Coverage

| Rubric Item | Status |
|-------------|--------|
| Code is functional and passes all tests | ✅ |
| Well-documented with docstrings | ✅ |
| EDA with correct summary statistics | ✅ |
| Rank-based recommendation functions | ✅ |
| User-item matrix creation | ✅ |
| Find similar users | ✅ |
| User-user collaborative filtering | ✅ |
| Improved CF with popularity ranking | ✅ |
| New user recommendations | ✅ |
| KMeans clustering on TF-IDF | ✅ |
| Content-based recommendation function | ✅ |
| SVD decomposition | ✅ |
| Latent feature selection with justification | ✅ |
| Article-article SVD recommendations | ✅ |
| Results discussion + A/B test proposal | ✅ |
