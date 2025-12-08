# Day 3: Matrix Operations  
# Working with 2D arrays of numbers
# Amey Prakash Sawant

import numpy as np

# Matrices - just rectangles of numbers
print("Day 3: Matrix Operations")
print("=" * 25)

# Create a matrix (2D list)
matrix_a = [[1, 2, 3],
            [4, 5, 6]]

matrix_b = [[7, 8, 9],
            [10, 11, 12]]

print("Matrix A:")
for row in matrix_a:
    print(row)

print("\nMatrix B:")  
for row in matrix_b:
    print(row)

# Matrix addition - add corresponding numbers
def add_matrices(mat1, mat2):
    result = []
    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat1[i])):
            new_row.append(mat1[i][j] + mat2[i][j])
        result.append(new_row)
    return result

sum_matrix = add_matrices(matrix_a, matrix_b)
print("\nA + B:")
for row in sum_matrix:
    print(row)

# Multiply by a number (scalar)
def multiply_by_number(matrix, num):
    result = []
    for i in range(len(matrix)):
        new_row = []
        for j in range(len(matrix[i])):
            new_row.append(matrix[i][j] * num)
        result.append(new_row)
    return result

scaled = multiply_by_number(matrix_a, 3)
print(f"\n3 * A:")
for row in scaled:
    print(row)

# Matrix multiplication - the tricky one!
print("\nMatrix Multiplication:")
print("-" * 21)

# For matrices A (2x3) and B (3x2)
# Result will be (2x2)
a_mat = [[1, 2, 3],
         [4, 5, 6]]  # 2 rows, 3 columns

b_mat = [[7,  8],
         [9,  10], 
         [11, 12]]   # 3 rows, 2 columns

def matrix_multiply(mat1, mat2):
    # mat1 is m×n, mat2 is n×p, result is m×p
    rows1 = len(mat1)
    cols1 = len(mat1[0])
    rows2 = len(mat2) 
    cols2 = len(mat2[0])
    
    # Check if multiplication is possible
    if cols1 != rows2:
        return "Can't multiply - dimensions don't match!"
    
    # Create result matrix
    result = []
    for i in range(rows1):
        new_row = []
        for j in range(cols2):
            # Calculate dot product of row i and column j
            dot_product = 0
            for k in range(cols1):
                dot_product += mat1[i][k] * mat2[k][j]
            new_row.append(dot_product)
        result.append(new_row)
    
    return result

result = matrix_multiply(a_mat, b_mat)
print("A (2x3):")
for row in a_mat:
    print(row)

print("\nB (3x2):")
for row in b_mat:
    print(row)

print("\nA × B (2x2):")
for row in result:
    print(row)

# Show one calculation step by step
print(f"\nCalculating result[0][0]:")
print(f"Row 0 of A: {a_mat[0]}")
print(f"Column 0 of B: {[b_mat[i][0] for i in range(len(b_mat))]}")
print(f"Dot product: {a_mat[0][0]}×{b_mat[0][0]} + {a_mat[0][1]}×{b_mat[1][0]} + {a_mat[0][2]}×{b_mat[2][0]}")
print(f"= {a_mat[0][0]*b_mat[0][0]} + {a_mat[0][1]*b_mat[1][0]} + {a_mat[0][2]*b_mat[2][0]} = {result[0][0]}")

# Matrix multiplication is NOT commutative!
print("\nOrder matters in multiplication:")
small_a = [[1, 2], [3, 4]]
small_b = [[5, 6], [7, 8]]

ab = matrix_multiply(small_a, small_b)
ba = matrix_multiply(small_b, small_a)

print("A × B:")
for row in ab:
    print(row)

print("\nB × A:")  
for row in ba:
    print(row)

print("A × B ≠ B × A (usually)")

# Identity matrix - like multiplying by 1
print("\nIdentity Matrix:")
print("-" * 15)

identity = [[1, 0, 0],
            [0, 1, 0], 
            [0, 0, 1]]

test_matrix = [[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]]

result_with_identity = matrix_multiply(test_matrix, identity)

print("Matrix × Identity = Matrix (unchanged)")
print("Original:")
for row in test_matrix:
    print(row)

print("\nAfter × Identity:")
for row in result_with_identity:
    print(row)

# Transpose - flip across diagonal
print("\nTranspose (flip across diagonal):")
print("-" * 32)

original = [[1, 2, 3],
            [4, 5, 6]]

def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    result = []
    
    for j in range(cols):  # New rows = old columns
        new_row = []
        for i in range(rows):  # New cols = old rows
            new_row.append(matrix[i][j])
        result.append(new_row)
    
    return result

transposed = transpose(original)

print("Original (2×3):")
for row in original:
    print(row)

print("\nTransposed (3×2):")
for row in transposed:
    print(row)

# Symmetric matrix - equals its transpose
symmetric = [[1, 2, 3],
             [2, 4, 5],
             [3, 5, 6]]

symmetric_T = transpose(symmetric)

print(f"\nSymmetric matrix example:")
print("Original:")
for row in symmetric:
    print(row)

print("\nTranspose:")
for row in symmetric_T:
    print(row)

print("They're the same! (symmetric)")

print("\nDay 3 complete! ✅")
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
