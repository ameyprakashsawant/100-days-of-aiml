# Project 3: Recommendation System Using Linear Algebra

Build a recommendation system using cosine similarity and matrix operations.

## 📚 Concepts Used

- Cosine Similarity
- Dot Product
- Matrix Factorization
- Euclidean Distance
- Normalized Vectors

## 🎯 Project Goals

1. Implement content-based filtering
2. Implement collaborative filtering
3. Build user-item matrix
4. Calculate similarity scores

## 🚀 How to Run

```bash
pip install numpy pandas
python recommendation_system.py
```

## 📖 Theory

### Cosine Similarity

$$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \cdot ||\mathbf{B}||}$$

### User-Item Matrix

```
         Item1  Item2  Item3
User1    5      3      0
User2    4      0      0
User3    1      1      0
User4    1      0      5
```

### Types of Filtering

1. **Content-Based**: Similar items based on features
2. **Collaborative**: Similar users' preferences
3. **Hybrid**: Combination of both
