# Day 4: Orthogonality & Orthogonal Matrices

## 📚 Learning Objectives

- Understand orthogonality and orthonormality
- Master orthogonal matrices and their properties
- Learn Gram-Schmidt orthogonalization
- Apply orthogonal projections

---

## 1. Orthogonality

### Definition

Two vectors u and v are **orthogonal** if their dot product is zero:
$$u \cdot v = 0$$

### Geometric Meaning

Orthogonal vectors are **perpendicular** (at 90° angles).

### Properties

- Zero vector is orthogonal to all vectors
- Orthogonal vectors are linearly independent (if non-zero)

---

## 2. Orthonormal Vectors

### Definition

Vectors are **orthonormal** if they are:

1. **Orthogonal** to each other: $u_i \cdot u_j = 0$ for $i \neq j$
2. **Normalized** (unit length): $\|u_i\| = 1$

### Example: Standard Basis

$$e_1 = [1, 0, 0], \quad e_2 = [0, 1, 0], \quad e_3 = [0, 0, 1]$$

---

## 3. Orthogonal Matrices

### Definition

A square matrix Q is **orthogonal** if:
$$Q^T Q = QQ^T = I$$

Equivalently: $Q^T = Q^{-1}$

### Properties

1. **Columns are orthonormal**
2. **Rows are orthonormal**
3. **Preserves lengths**: $\|Qx\| = \|x\|$
4. **Preserves angles**: $(Qx) \cdot (Qy) = x \cdot y$
5. **Determinant**: $\det(Q) = \pm 1$

### Geometric Interpretation

Orthogonal matrices represent:

- **Rotations** (det = +1)
- **Reflections** (det = -1)

---

## 4. Rotation Matrices

### 2D Rotation by angle θ

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

### Properties

- $R(\theta)^T = R(-\theta)$
- $R(\theta_1)R(\theta_2) = R(\theta_1 + \theta_2)$
- $R(\theta)^{-1} = R(-\theta) = R(\theta)^T$

---

## 5. Gram-Schmidt Orthogonalization

### Purpose

Convert any set of linearly independent vectors into an orthonormal set.

### Algorithm

Given vectors $v_1, v_2, ..., v_n$:

1. $u_1 = v_1 / \|v_1\|$

2. $u_2 = v_2 - (v_2 \cdot u_1)u_1$, then normalize

3. $u_k = v_k - \sum_{j=1}^{k-1}(v_k \cdot u_j)u_j$, then normalize

### Visual Idea

- Take each new vector
- Subtract its projections onto all previous orthonormal vectors
- Normalize the result

---

## 6. QR Decomposition

### Definition

Any matrix A can be decomposed as:
$$A = QR$$

Where:

- **Q**: Orthogonal matrix (columns are orthonormal)
- **R**: Upper triangular matrix

### Applications

- Solving least squares problems
- Eigenvalue algorithms
- More numerically stable than normal equations

---

## 7. Orthogonal Projections

### Projection onto a Subspace

To project vector b onto the column space of matrix A:

$$\hat{b} = A(A^TA)^{-1}A^Tb$$

If A has orthonormal columns:
$$\hat{b} = AA^Tb$$

### Projection Matrix

$$P = A(A^TA)^{-1}A^T$$

Properties:

- $P^2 = P$ (idempotent)
- $P^T = P$ (symmetric)

---

## 8. Applications in Machine Learning

| Concept               | ML Application                      |
| --------------------- | ----------------------------------- |
| Orthogonal matrices   | PCA transformations                 |
| QR decomposition      | Solving linear systems              |
| Gram-Schmidt          | Creating orthonormal features       |
| Orthogonal projection | Least squares regression            |
| Rotation matrices     | Data augmentation, image processing |

### Why Orthogonality Matters

1. **Numerical stability**: Orthogonal operations preserve precision
2. **Independence**: Orthogonal features capture unique information
3. **Efficiency**: Orthogonal matrices have trivial inverses (just transpose!)

---

## 🔑 Key Takeaways

| Concept           | Definition               | Key Property               |
| ----------------- | ------------------------ | -------------------------- |
| Orthogonal        | u·v = 0                  | Perpendicular              |
| Orthonormal       | Orthogonal + unit length | e1, e2, e3                 |
| Orthogonal Matrix | Q^T Q = I                | Q^-1 = Q^T                 |
| Gram-Schmidt      | Makes orthonormal set    | Subtract projections       |
| QR Decomposition  | A = QR                   | Q orthogonal, R triangular |

---

## 📖 Further Reading

- [3Blue1Brown: Change of basis](https://www.youtube.com/watch?v=P2LTAUO1TdA)
- Numerical Linear Algebra by Trefethen and Bau
