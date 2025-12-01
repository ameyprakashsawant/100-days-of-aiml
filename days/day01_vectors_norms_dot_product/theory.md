# Day 1: Vectors, Norms, Dot Product & Projections

## 📚 Learning Objectives

- Understand vectors and their representations
- Learn different types of norms
- Master dot product and its applications
- Understand vector projections

---

## 1. Vectors

### What is a Vector?

A vector is an ordered list of numbers that represents magnitude and direction in space.

```
v = [v₁, v₂, ..., vₙ]
```

### Types of Vectors

- **Row Vector**: 1 × n matrix `[1, 2, 3]`
- **Column Vector**: n × 1 matrix
- **Zero Vector**: All elements are 0
- **Unit Vector**: Vector with magnitude 1

### Vector Operations

- **Addition**: `u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]`
- **Scalar Multiplication**: `c·v = [c·v₁, c·v₂, ..., c·vₙ]`

---

## 2. Norms (Vector Magnitude)

Norms measure the "size" or "length" of a vector.

### L1 Norm (Manhattan Distance)

$$\|v\|_1 = \sum_{i=1}^{n} |v_i|$$

### L2 Norm (Euclidean Distance)

$$\|v\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}$$

### L∞ Norm (Max Norm)

$$\|v\|_\infty = \max_i |v_i|$$

### Lp Norm (General)

$$\|v\|_p = \left(\sum_{i=1}^{n} |v_i|^p\right)^{1/p}$$

### Applications in ML

- **L1 Regularization (Lasso)**: Promotes sparsity
- **L2 Regularization (Ridge)**: Prevents large weights
- **Distance metrics**: Measuring similarity between data points

---

## 3. Dot Product (Inner Product)

### Definition

$$u \cdot v = \sum_{i=1}^{n} u_i \cdot v_i = \|u\| \|v\| \cos(\theta)$$

### Properties

- **Commutative**: u·v = v·u
- **Distributive**: u·(v+w) = u·v + u·w
- **Scalar Multiplication**: (cu)·v = c(u·v)

### Geometric Interpretation

- If u·v > 0: vectors point in similar directions
- If u·v = 0: vectors are orthogonal (perpendicular)
- If u·v < 0: vectors point in opposite directions

### Applications in ML

- **Similarity measurement** (cosine similarity)
- **Neural network forward pass**
- **Attention mechanisms**

---

## 4. Vector Projections

### Scalar Projection

The length of the shadow of v onto u:
$$\text{comp}_u v = \frac{u \cdot v}{\|u\|}$$

### Vector Projection

The actual shadow vector:
$$\text{proj}_u v = \frac{u \cdot v}{\|u\|^2} u$$

### Applications in ML

- **Principal Component Analysis (PCA)**
- **Least squares regression**
- **Gram-Schmidt orthogonalization**

---

## 5. Cosine Similarity

$$\cos(\theta) = \frac{u \cdot v}{\|u\| \|v\|}$$

- Range: [-1, 1]
- 1 = identical direction
- 0 = orthogonal
- -1 = opposite direction

### Used in:

- Document similarity
- Recommendation systems
- Word embeddings (Word2Vec, GloVe)

---

## 🔑 Key Takeaways

| Concept           | Formula      | ML Application              |
| ----------------- | ------------ | --------------------------- |
| L2 Norm           | √(Σvᵢ²)      | Regularization, distance    |
| Dot Product       | Σuᵢvᵢ        | Neural networks, similarity |
| Projection        | (u·v/‖u‖²)u  | PCA, regression             |
| Cosine Similarity | u·v/(‖u‖‖v‖) | NLP, recommendations        |

---

## 📖 Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- Linear Algebra and Its Applications by Gilbert Strang
