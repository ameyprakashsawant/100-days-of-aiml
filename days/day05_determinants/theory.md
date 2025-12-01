# Day 5: Determinants

## 📚 Learning Objectives

- Understand what determinants represent geometrically
- Learn to compute determinants
- Master properties and applications of determinants

---

## 1. What is a Determinant?

### Definition

The **determinant** is a scalar value computed from a square matrix that encapsulates important properties of the matrix.

### Notation

$$\det(A) = |A|$$

### 2×2 Matrix

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

### 3×3 Matrix (Sarrus Rule)

$$\det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = aei + bfg + cdh - ceg - bdi - afh$$

---

## 2. Geometric Interpretation

### Key Insight

The determinant represents the **scaling factor** of the linear transformation.

### 2D: Area Scaling

- |det(A)| = factor by which area is scaled
- det(A) > 0: orientation preserved
- det(A) < 0: orientation flipped

### 3D: Volume Scaling

- |det(A)| = factor by which volume is scaled

### Examples

```
det([[2, 0], [0, 2]]) = 4  → Areas scaled by 4x
det([[1, 0], [0, 1]]) = 1  → No scaling (identity)
det([[1, 0], [0, 0]]) = 0  → Flattens to lower dimension
```

---

## 3. Properties of Determinants

### 1. Identity Matrix

$$\det(I) = 1$$

### 2. Transpose

$$\det(A^T) = \det(A)$$

### 3. Product Rule

$$\det(AB) = \det(A) \cdot \det(B)$$

### 4. Inverse

$$\det(A^{-1}) = \frac{1}{\det(A)}$$

### 5. Scalar Multiplication

$$\det(cA) = c^n \det(A)$$ for n×n matrix

### 6. Row Operations

- Swapping rows: det changes sign
- Multiplying row by c: det multiplied by c
- Adding rows: det unchanged

### 7. Triangular Matrices

$$\det = \text{product of diagonal elements}$$

---

## 4. Computing Determinants

### Cofactor Expansion

For any row i:
$$\det(A) = \sum_{j=1}^{n} a_{ij} \cdot C_{ij}$$

Where $C_{ij} = (-1)^{i+j} \cdot M_{ij}$ (cofactor)
and $M_{ij}$ is the minor (determinant of submatrix).

### LU Decomposition Method

1. Decompose A = LU
2. det(A) = det(L) × det(U)
3. For triangular matrices: det = product of diagonal

---

## 5. Determinant and Matrix Properties

| det(A) | Matrix A                 |
| ------ | ------------------------ |
| ≠ 0    | Invertible, full rank    |
| = 0    | Singular, not invertible |
| = 1    | Orthogonal (rotation)    |
| = -1   | Orthogonal (reflection)  |

---

## 6. Cramer's Rule

For solving Ax = b when det(A) ≠ 0:

$$x_i = \frac{\det(A_i)}{\det(A)}$$

Where $A_i$ is A with column i replaced by b.

### Practical Use

- Useful for small systems
- Inefficient for large systems (use LU instead)

---

## 7. Applications

### Machine Learning

1. **Checking invertibility** of covariance matrices
2. **Gaussian distributions**: det appears in normalization
3. **Change of variables** in probability

### Multivariate Gaussian

$$p(x) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$

The |Σ| (determinant) normalizes the distribution.

---

## 8. Eigenvalues and Determinant

### Relationship

$$\det(A) = \prod_{i=1}^{n} \lambda_i$$

The determinant equals the **product of all eigenvalues**.

### Characteristic Equation

$$\det(A - \lambda I) = 0$$

Used to find eigenvalues.

---

## 🔑 Key Takeaways

| Property   | Meaning                       |
| ---------- | ----------------------------- |
| det(A) ≠ 0 | A is invertible               |
| det(A) = 0 | A is singular                 |
| \|det(A)\| | Area/volume scaling factor    |
| sign(det)  | Orientation preserved/flipped |
| det(AB)    | = det(A) × det(B)             |

---

## 📖 Further Reading

- [3Blue1Brown: The determinant](https://www.youtube.com/watch?v=Ip3X9LOh2dk)
- Matrix Analysis and Applied Linear Algebra by Carl Meyer
