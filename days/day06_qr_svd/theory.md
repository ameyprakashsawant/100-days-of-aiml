# Day 6: QR Decomposition & SVD

## 📚 Learning Objectives

- Master QR decomposition and its applications
- Understand Singular Value Decomposition (SVD)
- Learn to apply these decompositions in ML

---

## 1. QR Decomposition

### Definition

Any matrix A (m×n) can be decomposed as:
$$A = QR$$

Where:

- **Q** (m×m): Orthogonal matrix
- **R** (m×n): Upper triangular matrix

### Computation Methods

1. **Gram-Schmidt**: Classic but numerically unstable
2. **Modified Gram-Schmidt**: More stable
3. **Householder reflections**: Most stable, O(2mn² - 2n³/3)
4. **Givens rotations**: Good for sparse matrices

### Properties

- Q has orthonormal columns
- R is upper triangular
- If A has full column rank, R has positive diagonal entries

---

## 2. QR Applications

### Solving Least Squares

Instead of normal equations: $\theta = (X^TX)^{-1}X^Ty$

Use QR:
$$X = QR$$
$$R\theta = Q^Ty$$

Then solve by back substitution (R is triangular).

### Advantages

- Numerically more stable than normal equations
- Avoids computing $X^TX$
- Better for ill-conditioned matrices

### Eigenvalue Computation (QR Algorithm)

1. Start with A₀ = A
2. Compute Aₖ = QₖRₖ
3. Set Aₖ₊₁ = RₖQₖ
4. Repeat until convergence
5. Diagonal of final A contains eigenvalues

---

## 3. Singular Value Decomposition (SVD)

### Definition

Any matrix A (m×n) can be decomposed as:
$$A = U\Sigma V^T$$

Where:

- **U** (m×m): Left singular vectors (orthogonal)
- **Σ** (m×n): Diagonal matrix of singular values
- **V** (n×n): Right singular vectors (orthogonal)

### Visualization

```
A      =    U    ×    Σ    ×    V^T
(m×n)     (m×m)     (m×n)     (n×n)
```

### Singular Values

$$\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_r > 0$$

where r = rank(A)

---

## 4. Geometric Interpretation of SVD

### The Transformation Ax can be viewed as:

1. **V^T**: Rotate in input space
2. **Σ**: Scale along axes
3. **U**: Rotate in output space

### Key Insight

- Columns of V: Orthonormal basis for input space
- Columns of U: Orthonormal basis for output space
- Singular values: Amount of stretching along each axis

---

## 5. Properties of SVD

### Relationship to Eigenvalues

- Singular values of A = √(eigenvalues of AᵀA)
- Columns of U = eigenvectors of AAᵀ
- Columns of V = eigenvectors of AᵀA

### Matrix Properties from SVD

- **Rank**: Number of non-zero singular values
- **Null space**: Columns of V for zero singular values
- **Range**: Columns of U for non-zero singular values
- **2-Norm**: Largest singular value σ₁
- **Frobenius Norm**: √(Σσᵢ²)
- **Condition number**: σ₁/σᵣ

---

## 6. Truncated SVD (Low-Rank Approximation)

### Best Rank-k Approximation

$$A_k = U_k \Sigma_k V_k^T$$

Keep only top k singular values.

### Eckart-Young Theorem

$A_k$ is the closest rank-k matrix to A (in both 2-norm and Frobenius norm).

### Error

$$\|A - A_k\|_2 = \sigma_{k+1}$$

---

## 7. Applications in Machine Learning

### 1. Dimensionality Reduction (PCA)

- Center data: X̄ = X - mean
- Compute SVD: X̄ = UΣVᵀ
- Principal components = columns of V
- Project: X_reduced = X̄V_k

### 2. Recommender Systems

- User-item matrix ≈ UΣVᵀ
- Collaborative filtering
- Netflix Prize winning solution

### 3. Image Compression

- Each image = matrix of pixels
- Keep top k singular values
- Compression ratio: k(m+n+1)/(mn)

### 4. Latent Semantic Analysis (LSA)

- Document-term matrix ≈ UΣVᵀ
- Captures semantic relationships
- Used in information retrieval

### 5. Pseudoinverse

$$A^+ = V\Sigma^+U^T$$

where Σ⁺ has 1/σᵢ on diagonal (for non-zero σᵢ)

---

## 8. Comparing QR and SVD

| Aspect    | QR                     | SVD                        |
| --------- | ---------------------- | -------------------------- |
| Cost      | O(mn²)                 | O(mn²) but larger constant |
| Use case  | Solving linear systems | Low-rank approximation     |
| Output    | Q, R                   | U, Σ, V                    |
| Stability | Very stable            | Very stable                |

---

## 🔑 Key Takeaways

| Decomposition | Formula     | Key Application            |
| ------------- | ----------- | -------------------------- |
| QR            | A = QR      | Least squares, eigenvalues |
| SVD           | A = UΣVᵀ    | Dimensionality reduction   |
| Truncated SVD | A ≈ UₖΣₖVₖᵀ | Compression, denoising     |

---

## 📖 Further Reading

- [Steve Brunton: SVD](https://www.youtube.com/watch?v=gXbThCXjZFM)
- Numerical Linear Algebra by Trefethen and Bau
- Mining of Massive Datasets (Chapter 11)
