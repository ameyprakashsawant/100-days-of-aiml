"""
Day 3: Matrix Operations
Practical Implementation in Python

Author: Amey Prakash Sawant
100 Days of AI/ML Journey
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. MATRIX CREATION
# ============================================

print("=" * 50)
print("1. MATRIX CREATION")
print("=" * 50)

# Different ways to create matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[7, 8, 9],
              [10, 11, 12]])

print(f"Matrix A (2x3):\n{A}")
print(f"\nMatrix B (2x3):\n{B}")

# Special matrices
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
identity = np.eye(4)
diagonal = np.diag([1, 2, 3, 4])
random_matrix = np.random.randn(3, 3)

print(f"\nZero matrix (3x3):\n{zeros}")
print(f"\nIdentity matrix (4x4):\n{identity}")
print(f"\nDiagonal matrix:\n{diagonal}")


# ============================================
# 2. MATRIX ADDITION AND SCALAR OPERATIONS
# ============================================

print("\n" + "=" * 50)
print("2. MATRIX ADDITION AND SCALAR OPERATIONS")
print("=" * 50)

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Addition
C = A + B
print(f"A:\n{A}")
print(f"B:\n{B}")
print(f"A + B:\n{C}")

# Subtraction
D = A - B
print(f"\nA - B:\n{D}")

# Scalar multiplication
scalar = 3
E = scalar * A
print(f"\n3 * A:\n{E}")

# Element-wise operations
F = A * B  # Hadamard product (element-wise)
print(f"\nA * B (element-wise):\n{F}")


# ============================================
# 3. MATRIX MULTIPLICATION
# ============================================

print("\n" + "=" * 50)
print("3. MATRIX MULTIPLICATION")
print("=" * 50)

A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])  # 3x2

# Matrix multiplication
C = A @ B  # or np.matmul(A, B) or np.dot(A, B)
print(f"A (2x3):\n{A}")
print(f"\nB (3x2):\n{B}")
print(f"\nA @ B (2x2):\n{C}")

# Verify dimensions
print(f"\nDimensions: {A.shape} @ {B.shape} = {C.shape}")

# Manual calculation verification
print("\nManual verification of C[0,0]:")
print(f"  A[0,:] @ B[:,0] = {A[0,:]} @ {B[:,0]} = {np.dot(A[0,:], B[:,0])}")

# Matrix multiplication is NOT commutative
print("\n--- Commutativity Check ---")
P = np.array([[1, 2],
              [3, 4]])
Q = np.array([[5, 6],
              [7, 8]])
print(f"P @ Q:\n{P @ Q}")
print(f"\nQ @ P:\n{Q @ P}")
print(f"\nP @ Q == Q @ P? {np.allclose(P @ Q, Q @ P)}")


# ============================================
# 4. IDENTITY MATRIX
# ============================================

print("\n" + "=" * 50)
print("4. IDENTITY MATRIX")
print("=" * 50)

I = np.eye(3)
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"Identity matrix I:\n{I}")
print(f"\nMatrix A:\n{A}")
print(f"\nA @ I:\n{A @ I}")
print(f"\nI @ A:\n{I @ A}")
print(f"\nA @ I == A? {np.allclose(A @ I, A)}")
print(f"I @ A == A? {np.allclose(I @ A, A)}")


# ============================================
# 5. TRANSPOSE
# ============================================

print("\n" + "=" * 50)
print("5. TRANSPOSE")
print("=" * 50)

A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(f"A:\n{A}")
print(f"A.shape: {A.shape}")

print(f"\nA.T (transpose):\n{A.T}")
print(f"A.T.shape: {A.T.shape}")

# Property: (A^T)^T = A
print(f"\n(A^T)^T == A? {np.allclose(A.T.T, A)}")

# Property: (AB)^T = B^T @ A^T
B = np.array([[1, 2],
              [3, 4],
              [5, 6]])

AB = A @ B
print(f"\n(A @ B)^T:\n{(A @ B).T}")
print(f"B^T @ A^T:\n{B.T @ A.T}")
print(f"(AB)^T == B^T @ A^T? {np.allclose((A @ B).T, B.T @ A.T)}")


# ============================================
# 6. SYMMETRIC MATRICES
# ============================================

print("\n" + "=" * 50)
print("6. SYMMETRIC MATRICES")
print("=" * 50)

# Create symmetric matrix
A = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])

print(f"Matrix A:\n{A}")
print(f"\nA^T:\n{A.T}")
print(f"\nIs A symmetric (A == A^T)? {np.allclose(A, A.T)}")

# Creating symmetric from any matrix: (A + A^T) / 2
B = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B_symmetric = (B + B.T) / 2
print(f"\nOriginal B:\n{B}")
print(f"\n(B + B^T)/2 (made symmetric):\n{B_symmetric}")
print(f"Is symmetric? {np.allclose(B_symmetric, B_symmetric.T)}")

# Covariance matrix is always symmetric
X = np.random.randn(100, 3)  # 100 samples, 3 features
cov_matrix = np.cov(X.T)
print(f"\nCovariance matrix:\n{cov_matrix}")
print(f"Is covariance symmetric? {np.allclose(cov_matrix, cov_matrix.T)}")


# ============================================
# 7. MATRIX INVERSE
# ============================================

print("\n" + "=" * 50)
print("7. MATRIX INVERSE")
print("=" * 50)

A = np.array([[4, 7],
              [2, 6]])

# Compute inverse
A_inv = np.linalg.inv(A)
print(f"Matrix A:\n{A}")
print(f"\nA inverse:\n{A_inv}")

# Verify: A @ A^-1 = I
product = A @ A_inv
print(f"\nA @ A^-1:\n{product}")
print(f"A @ A^-1 ≈ I? {np.allclose(product, np.eye(2))}")

# Check if matrix is invertible using determinant
det_A = np.linalg.det(A)
print(f"\nDeterminant of A: {det_A}")
print(f"Is A invertible (det ≠ 0)? {det_A != 0}")

# Singular matrix (not invertible)
singular = np.array([[1, 2],
                     [2, 4]])  # Row 2 = 2 * Row 1
det_singular = np.linalg.det(singular)
print(f"\nSingular matrix:\n{singular}")
print(f"Determinant: {det_singular:.10f}")
print("(Near zero, so not invertible)")


# ============================================
# 8. SOLVING LINEAR SYSTEMS
# ============================================

print("\n" + "=" * 50)
print("8. SOLVING LINEAR SYSTEMS (Ax = b)")
print("=" * 50)

# System of equations:
# 2x + 3y = 8
# 4x + 5y = 14

A = np.array([[2, 3],
              [4, 5]])
b = np.array([8, 14])

# Method 1: Using inverse (not recommended for large systems)
x_inv = np.linalg.inv(A) @ b
print(f"Solution using A^-1 @ b: x={x_inv[0]}, y={x_inv[1]}")

# Method 2: Using solve (recommended - more numerically stable)
x_solve = np.linalg.solve(A, b)
print(f"Solution using np.linalg.solve: x={x_solve[0]}, y={x_solve[1]}")

# Verify
print(f"\nVerification: A @ x = {A @ x_solve}")
print(f"Original b: {b}")


# ============================================
# 9. ML APPLICATION: Linear Regression
# ============================================

print("\n" + "=" * 50)
print("9. ML APPLICATION: Linear Regression")
print("=" * 50)

# Generate synthetic data
np.random.seed(42)
n_samples = 100
X_raw = np.random.randn(n_samples, 2)  # 2 features
true_weights = np.array([3, -2])
noise = np.random.randn(n_samples) * 0.5
y = X_raw @ true_weights + noise

# Add bias column
X = np.column_stack([np.ones(n_samples), X_raw])

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Normal equation: θ = (X^T X)^-1 X^T y
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
Xty = X.T @ y
theta = XtX_inv @ Xty

print(f"\nSolved weights: {theta}")
print(f"True weights (with bias=0): [0, {true_weights[0]}, {true_weights[1]}]")

# Using np.linalg.lstsq (more numerically stable)
theta_lstsq, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
print(f"\nWeights using lstsq: {theta_lstsq}")


# ============================================
# 10. ML APPLICATION: Neural Network Layer
# ============================================

print("\n" + "=" * 50)
print("10. ML APPLICATION: Neural Network Layer")
print("=" * 50)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Simple neural network layer
input_size = 4
hidden_size = 3
output_size = 2

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros(output_size)

# Sample input (batch of 2 samples)
X = np.array([[1, 2, 3, 4],
              [0.5, 1.5, 2.5, 3.5]])

print(f"Input X shape: {X.shape}")
print(f"W1 shape: {W1.shape}")
print(f"W2 shape: {W2.shape}")

# Forward pass
z1 = X @ W1 + b1  # Linear transformation
a1 = relu(z1)     # Activation
print(f"\nLayer 1 output (after ReLU):\n{a1}")

z2 = a1 @ W2 + b2  # Linear transformation
output = sigmoid(z2)  # Final activation
print(f"\nFinal output (after sigmoid):\n{output}")


# ============================================
# 11. VISUALIZATION
# ============================================

print("\n" + "=" * 50)
print("11. VISUALIZATION")
print("=" * 50)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Matrix multiplication visualization
ax1 = axes[0]
A = np.random.rand(4, 3)
B = np.random.rand(3, 5)
C = A @ B

ax1.matshow(np.hstack([A, np.ones((4, 1))*np.nan, B.T, 
                       np.ones((4, 1))*np.nan, C]), cmap='viridis')
ax1.set_title('Matrix Multiplication: A(4x3) @ B(3x5) = C(4x5)')
ax1.axis('off')

# Plot 2: Identity matrix property
ax2 = axes[1]
I = np.eye(4)
ax2.matshow(I, cmap='Blues')
ax2.set_title('Identity Matrix (4x4)')
for i in range(4):
    for j in range(4):
        ax2.text(j, i, f'{int(I[i,j])}', ha='center', va='center', fontsize=14)

# Plot 3: Transpose visualization
ax3 = axes[2]
A = np.arange(1, 7).reshape(2, 3)
ax3.matshow(A, cmap='Greens')
ax3.set_title(f'Original Matrix (2x3)')
for i in range(2):
    for j in range(3):
        ax3.text(j, i, f'{A[i,j]}', ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.savefig('matrix_operations_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'matrix_operations_visualization.png'")


print("\n" + "=" * 50)
print("Day 3 Complete! 🎉")
print("=" * 50)
