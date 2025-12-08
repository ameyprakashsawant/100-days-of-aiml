# Day 7: Geometric Interpretations of Linear Algebra

## 📚 Learning Objectives

- Visualize linear transformations geometrically
- Understand eigenvalues/eigenvectors as special directions
- Connect linear algebra concepts to ML intuitively

---

## 1. Vectors as Points and Arrows

### Dual Interpretation

- **Arrow**: Direction + magnitude from origin
- **Point**: Location in space

### In ML Context

- Feature vectors = points in feature space
- Gradient = direction of steepest ascent
- Weight updates = arrows showing change

---

## 2. Matrices as Transformations

### Every Matrix is a Function

$$T(x) = Ax$$

Transforms input vector x to output vector Ax.

### Types of Transformations

| Matrix Type | Transformation        | Example                                                                             |
| ----------- | --------------------- | ----------------------------------------------------------------------------------- |
| Rotation    | Rotates around origin | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ |
| Scaling     | Stretches/compresses  | $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$                                  |
| Shear       | Slants shape          | $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$                                      |
| Reflection  | Mirrors across axis   | $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$                                     |
| Projection  | Flattens to subspace  | $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$                                      |

---

## 3. Column Space Visualization

### Definition

The **column space** of A is all possible outputs Ax.

### Geometric Meaning

- Span of column vectors
- "Reachable" space via linear transformation
- In 2D: could be point, line, or whole plane

### Rank Geometry

- Rank 0: Only origin reachable
- Rank 1: Line reachable
- Rank 2: Plane reachable
- Rank n: All of ℝⁿ reachable

---

## 4. Null Space Visualization

### Definition

The **null space** of A is all x where Ax = 0.

### Geometric Meaning

- Vectors that get "crushed" to zero
- The "forgotten" information
- Perpendicular to row space

### Dimension Relationship

$$\dim(\text{column space}) + \dim(\text{null space}) = n$$

---

## 5. Eigenvalues and Eigenvectors

### Definition

For matrix A, if:
$$Av = \lambda v$$

Then v is an **eigenvector** and λ is its **eigenvalue**.

### Geometric Interpretation

- Eigenvectors: Directions that only get scaled (not rotated)
- Eigenvalues: The scaling factors

### Visual Example

Under transformation A:

- Most vectors change direction
- Eigenvectors only stretch/shrink
- They're the "principal axes" of the transformation

---

## 6. Eigendecomposition

### Formula

$$A = V \Lambda V^{-1}$$

Where:

- V: Matrix of eigenvectors
- Λ: Diagonal matrix of eigenvalues

### Geometric Meaning

1. **V⁻¹**: Change to eigenvector basis
2. **Λ**: Scale along eigenvector axes
3. **V**: Change back to standard basis

### Powers of Matrices

$$A^n = V \Lambda^n V^{-1}$$

Why eigenvalues matter for stability!

---

## 7. SVD Geometry

### The Three Steps of Ax = UΣVᵀx

1. **Vᵀx**: Rotate to align with input singular vectors
2. **Σ**: Scale differently along each axis
3. **U**: Rotate to align with output singular vectors

### Unit Sphere Transformation

- Input: Unit sphere
- Output: Ellipsoid
- Semi-axes aligned with left singular vectors
- Semi-axis lengths = singular values

---

## 8. Projections

### Orthogonal Projection onto Line

$$\text{proj}_a(b) = \frac{a \cdot b}{\|a\|^2} a$$

### Geometric View

- Shadow of b onto line spanned by a
- Closest point to b on the line
- Error (b - proj) is perpendicular to a

### Projection onto Subspace

$$\hat{b} = A(A^TA)^{-1}A^Tb$$

---

## 9. Least Squares Geometry

### The Problem

Find x that minimizes ||Ax - b||

### Geometric View

1. b might not be in column space of A
2. Find closest point to b in column space
3. This is the projection of b onto column space
4. Residual (b - Ax̂) is perpendicular to column space

### Normal Equations

$$A^TAx = A^Tb$$

Says: "Make residual orthogonal to column space"

---

## 10. Applications in ML

### PCA Geometry

1. Data points form a cloud
2. Find directions of maximum variance
3. These are eigenvectors of covariance matrix
4. Project data onto top k eigenvectors

### Neural Network Geometry

- Each layer: Linear transformation + nonlinear activation
- Weights matrices transform feature space
- Deep networks = composition of transformations

### Gradient Descent Geometry

- Gradient points "uphill"
- Negative gradient is steepest descent direction
- Learning rate scales the step size

### Regularization Geometry

- L2: Keep weights in a ball
- L1: Keep weights in a diamond (promotes sparsity)
- Constraint surface intersects loss surface

---

## 🔑 Key Takeaways

| Concept       | Geometric View              |
| ------------- | --------------------------- |
| Matrix        | Transformation              |
| Column space  | Reachable outputs           |
| Null space    | "Crushed" inputs            |
| Eigenvector   | Direction preserved         |
| Eigenvalue    | Scaling factor              |
| SVD           | Rotation → Scale → Rotation |
| Projection    | Shadow onto subspace        |
| Least squares | Project b onto column space |

---

## 📖 Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- Visual Linear Algebra by Harry Porter
- Linear Algebra Done Right by Sheldon Axler
