# Day 2: Span, Linear Independence & Basis

## 📚 Learning Objectives

- Understand what span means geometrically
- Master linear independence and dependence
- Learn about basis vectors and their importance
- Understand vector spaces

---

## 1. Span

### Definition

The **span** of a set of vectors is all possible linear combinations of those vectors.

$$\text{span}(\{v_1, v_2, ..., v_k\}) = \{c_1v_1 + c_2v_2 + ... + c_kv_k : c_i \in \mathbb{R}\}$$

### Geometric Interpretation

- **1 non-zero vector**: Span is a line through origin
- **2 non-parallel vectors in ℝ²**: Span is entire plane
- **2 non-parallel vectors in ℝ³**: Span is a plane in 3D space
- **3 non-coplanar vectors in ℝ³**: Span is entire 3D space

### Example

```
v₁ = [1, 0]
v₂ = [0, 1]
span({v₁, v₂}) = ℝ² (the entire 2D plane)
```

---

## 2. Linear Independence

### Definition

Vectors {v₁, v₂, ..., vₖ} are **linearly independent** if:

$$c_1v_1 + c_2v_2 + ... + c_kv_k = 0$$

has ONLY the trivial solution: c₁ = c₂ = ... = cₖ = 0

### Linear Dependence

If there exist non-zero scalars that satisfy the equation above, the vectors are **linearly dependent**.

### Intuition

- **Independent**: No vector can be written as a combination of others
- **Dependent**: At least one vector is "redundant"

### Quick Tests

1. **Two vectors**: Dependent if one is a scalar multiple of the other
2. **More vectors than dimensions**: Always dependent
3. **Contains zero vector**: Always dependent

### Example

```
Independent: v₁ = [1, 0], v₂ = [0, 1]
Dependent: v₁ = [1, 2], v₂ = [2, 4] (v₂ = 2·v₁)
```

---

## 3. Basis

### Definition

A **basis** of a vector space V is a set of vectors that:

1. **Spans** V (can reach any vector in V)
2. Is **linearly independent** (minimal, no redundancy)

### Standard Basis for ℝⁿ

```
ℝ²: e₁ = [1, 0], e₂ = [0, 1]
ℝ³: e₁ = [1, 0, 0], e₂ = [0, 1, 0], e₃ = [0, 0, 1]
```

### Properties

- All bases of a vector space have the same number of vectors
- This number is called the **dimension** of the space
- Any vector in the space can be uniquely represented using basis vectors

### Change of Basis

Any vector can be expressed in different bases:

```
v = 3e₁ + 2e₂ (standard basis)
v = 1b₁ + 5b₂ (different basis)
```

---

## 4. Vector Spaces

### Definition

A **vector space** is a set V with two operations (addition and scalar multiplication) that satisfy:

1. Closure under addition
2. Closure under scalar multiplication
3. Commutativity of addition
4. Associativity
5. Existence of zero vector
6. Existence of additive inverses
7. Distributive properties
8. Scalar multiplication identity

### Common Vector Spaces

- ℝⁿ (n-dimensional real numbers)
- Polynomials of degree ≤ n
- m × n matrices
- Continuous functions

---

## 5. Dimension

### Definition

The **dimension** of a vector space is the number of vectors in any basis.

$$\dim(\mathbb{R}^n) = n$$

### Important Facts

- dim(ℝ²) = 2
- dim(ℝ³) = 3
- For an m×n matrix space: dim = m×n

---

## 6. Rank

### Definition

The **rank** of a matrix is:

- The dimension of its column space
- The dimension of its row space
- The number of linearly independent columns (or rows)

### Properties

- rank(A) ≤ min(m, n) for m×n matrix
- If rank(A) = n, columns are linearly independent
- If rank(A) = m, rows are linearly independent

---

## 7. Applications in Machine Learning

### Why This Matters

| Concept             | ML Application                                |
| ------------------- | --------------------------------------------- |
| Span                | Feature space representation                  |
| Linear Independence | Avoiding redundant features                   |
| Basis               | PCA principal components                      |
| Rank                | Matrix factorization, low-rank approximations |
| Dimension           | Intrinsic dimensionality of data              |

### Dimensionality Reduction

- High-dimensional data often lies on a lower-dimensional subspace
- PCA finds a new basis that captures most variance
- The span of principal components defines the reduced space

### Feature Engineering

- Linearly dependent features provide no new information
- Check for multicollinearity in regression

---

## 🔑 Key Takeaways

1. **Span** = all vectors reachable by linear combinations
2. **Linear Independence** = no redundancy, no vector is a combo of others
3. **Basis** = minimal spanning set (independent + spans the space)
4. **Dimension** = number of vectors in any basis
5. **Rank** = number of independent columns/rows

---

## 📖 Further Reading

- [3Blue1Brown: Linear combinations, span, and basis vectors](https://www.youtube.com/watch?v=k7RM-ot2NWY)
- Gilbert Strang's Linear Algebra Lectures
