# Project 2: PCA From Scratch

Implementing Principal Component Analysis using pure linear algebra.

## 📚 Concepts Used

- Eigenvalues & Eigenvectors
- Covariance Matrix
- SVD (Singular Value Decomposition)
- Matrix Multiplication
- Variance

## 🎯 Project Goals

1. Implement PCA from scratch (no sklearn)
2. Visualize dimensionality reduction
3. Compare with sklearn's PCA
4. Apply to real dataset

## 🚀 How to Run

```bash
pip install numpy matplotlib
python pca_from_scratch.py
```

## 📖 Theory

### PCA Algorithm Steps:

1. **Center the data**: X̄ = X - mean(X)
2. **Compute covariance matrix**: C = (1/n) X̄ᵀX̄
3. **Compute eigenvectors/values**: Cv = λv
4. **Sort by eigenvalue** (descending)
5. **Project**: X_reduced = X̄ @ V[:, :k]

### Alternative: SVD Approach

1. Center data: X̄
2. Compute SVD: X̄ = UΣVᵀ
3. Principal components are columns of V
4. Singular values² / n = Eigenvalues

### Explained Variance

$$\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$
