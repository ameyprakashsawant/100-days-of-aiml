# Day 3: Matrix Operations

## 📚 Learning Objectives

- Master matrix addition and multiplication
- Understand identity and transpose matrices
- Learn matrix properties and their importance in ML

---

## 1. Matrix Basics

### What is a Matrix?

A matrix is a 2D array of numbers arranged in rows and columns.

$$A_{m \times n} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

### Special Matrices

- **Square Matrix**: m = n
- **Row Vector**: 1 × n matrix
- **Column Vector**: m × 1 matrix
- **Zero Matrix**: All elements are 0
- **Diagonal Matrix**: Non-zero elements only on main diagonal

---

## 2. Matrix Addition

### Rules

- Matrices must have the **same dimensions**
- Add corresponding elements

$$A + B = \begin{bmatrix} a_{11}+b_{11} & a_{12}+b_{12} \\ a_{21}+b_{21} & a_{22}+b_{22} \end{bmatrix}$$

### Properties

- **Commutative**: A + B = B + A
- **Associative**: (A + B) + C = A + (B + C)
- **Identity**: A + 0 = A

---

## 3. Scalar Multiplication

$$cA = \begin{bmatrix} ca_{11} & ca_{12} \\ ca_{21} & ca_{22} \end{bmatrix}$$

### Properties

- **Distributive**: c(A + B) = cA + cB
- **(c + d)A = cA + dA**
- **(cd)A = c(dA)**

---

## 4. Matrix Multiplication

### Definition

For A (m×n) and B (n×p), the product C = AB is (m×p):

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

### Key Requirements

- **Number of columns in A** = **Number of rows in B**
- Result dimensions: (m×n) × (n×p) = (m×p)

### Visualization

```
[m×n] × [n×p] = [m×p]
        ↑
    Must match!
```

### ⚠️ NOT Commutative!

$$AB \neq BA$$ (in general)

### Properties

- **Associative**: (AB)C = A(BC)
- **Distributive**: A(B + C) = AB + AC
- **NOT Commutative**: AB ≠ BA

---

## 5. Identity Matrix

### Definition

Square matrix with 1s on diagonal, 0s elsewhere:

$$I_n = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

### Property

$$AI = IA = A$$

---

## 6. Matrix Transpose

### Definition

Swap rows and columns:

$$A^T_{ij} = A_{ji}$$

### Example

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \Rightarrow A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

### Properties

- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$
- $(AB)^T = B^T A^T$ ⚡ (order reverses!)
- $(cA)^T = cA^T$

---

## 7. Symmetric Matrices

### Definition

A matrix is symmetric if:
$$A = A^T$$

### Properties

- All eigenvalues are real
- Eigenvectors are orthogonal
- Common in covariance matrices

### Example

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{bmatrix}$$

---

## 8. Matrix Inverse

### Definition

For square matrix A, its inverse $A^{-1}$ satisfies:
$$AA^{-1} = A^{-1}A = I$$

### Properties

- Not all matrices have inverses
- $(AB)^{-1} = B^{-1}A^{-1}$ (order reverses!)
- $(A^T)^{-1} = (A^{-1})^T$
- $(A^{-1})^{-1} = A$

### Conditions for Invertibility

- Matrix must be square
- Determinant ≠ 0
- Full rank (all columns linearly independent)

---

## 9. Applications in Machine Learning

| Operation       | ML Application                       |
| --------------- | ------------------------------------ |
| Matrix Multiply | Neural network forward pass          |
| Transpose       | Gradient computation                 |
| Inverse         | Solving normal equations             |
| Identity        | Regularization (I added to matrices) |
| Symmetric       | Covariance matrices                  |

### Neural Network Layer

$$\text{output} = \sigma(Wx + b)$$

Where:

- W is weight matrix
- x is input vector
- b is bias vector
- σ is activation function

### Normal Equation (Linear Regression)

$$\theta = (X^TX)^{-1}X^Ty$$

---

## 🔑 Key Takeaways

| Operation      | Symbol | Key Point                   |
| -------------- | ------ | --------------------------- |
| Addition       | A + B  | Same dimensions required    |
| Multiplication | AB     | Inner dimensions must match |
| Transpose      | A^T    | Rows ↔ Columns              |
| Inverse        | A^{-1} | AA^{-1} = I                 |
| Identity       | I      | Multiplicative identity     |

---

## 📖 Further Reading

- [3Blue1Brown: Matrix multiplication as composition](https://www.youtube.com/watch?v=XkY2DOUCWMU)
- Linear Algebra Done Right by Sheldon Axler
