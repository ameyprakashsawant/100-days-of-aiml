"""
Day 2: Span, Linear Independence & Basis
Practical Implementation in Python

Author: Amey Prakash Sawant
100 Days of AI/ML Journey
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================
# 1. LINEAR COMBINATIONS AND SPAN
# ============================================

print("=" * 50)
print("1. LINEAR COMBINATIONS AND SPAN")
print("=" * 50)

# Define basis vectors
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Any vector in R² can be written as a linear combination
def linear_combination(vectors, coefficients):
    """Compute linear combination of vectors with given coefficients"""
    result = np.zeros(vectors[0].shape)
    for v, c in zip(vectors, coefficients):
        result += c * v
    return result

# Example: Create vector [3, 4] using basis vectors
target = np.array([3, 4])
coeffs = [3, 4]  # 3*v1 + 4*v2
result = linear_combination([v1, v2], coeffs)
print(f"v1 = {v1}, v2 = {v2}")
print(f"3*v1 + 4*v2 = {result}")
print(f"Target: {target}, Match: {np.allclose(result, target)}")

# Non-standard basis
b1 = np.array([1, 1])
b2 = np.array([1, -1])
# To get [3, 4] in this basis: solve c1*b1 + c2*b2 = [3, 4]
# [1, 1]*c1 + [1, -1]*c2 = [3, 4]
# c1 + c2 = 3
# c1 - c2 = 4
# c1 = 3.5, c2 = -0.5
coeffs_new = [3.5, -0.5]
result_new = linear_combination([b1, b2], coeffs_new)
print(f"\nUsing different basis:")
print(f"b1 = {b1}, b2 = {b2}")
print(f"3.5*b1 + (-0.5)*b2 = {result_new}")


# ============================================
# 2. CHECKING LINEAR INDEPENDENCE
# ============================================

print("\n" + "=" * 50)
print("2. CHECKING LINEAR INDEPENDENCE")
print("=" * 50)

def check_linear_independence(vectors):
    """
    Check if vectors are linearly independent using matrix rank.
    Returns True if independent, False if dependent.
    """
    # Stack vectors as columns of a matrix
    matrix = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(matrix)
    num_vectors = len(vectors)
    
    is_independent = rank == num_vectors
    return is_independent, rank, num_vectors

# Example 1: Independent vectors
v1 = np.array([1, 0])
v2 = np.array([0, 1])
is_ind, rank, n = check_linear_independence([v1, v2])
print(f"Vectors: {v1}, {v2}")
print(f"Independent: {is_ind} (rank={rank}, vectors={n})")

# Example 2: Dependent vectors
v1 = np.array([1, 2])
v2 = np.array([2, 4])  # v2 = 2*v1
is_ind, rank, n = check_linear_independence([v1, v2])
print(f"\nVectors: {v1}, {v2}")
print(f"Independent: {is_ind} (rank={rank}, vectors={n})")

# Example 3: 3 vectors in R²
v1 = np.array([1, 0])
v2 = np.array([0, 1])
v3 = np.array([1, 1])
is_ind, rank, n = check_linear_independence([v1, v2, v3])
print(f"\nVectors: {v1}, {v2}, {v3}")
print(f"Independent: {is_ind} (rank={rank}, vectors={n})")
print("(More vectors than dimensions → always dependent)")

# Example 4: 3D vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
is_ind, rank, n = check_linear_independence([v1, v2, v3])
print(f"\nVectors in 3D: {v1}, {v2}, {v3}")
print(f"Independent: {is_ind} (rank={rank}, vectors={n})")


# ============================================
# 3. FINDING BASIS FROM A SET OF VECTORS
# ============================================

print("\n" + "=" * 50)
print("3. FINDING BASIS FROM A SET OF VECTORS")
print("=" * 50)

def find_basis(vectors):
    """
    Find a basis from a set of vectors using row reduction.
    Returns indices of vectors that form a basis.
    """
    matrix = np.column_stack(vectors)
    m, n = matrix.shape
    
    # Use QR decomposition to find independent columns
    Q, R = np.linalg.qr(matrix)
    
    # Independent columns have non-zero diagonal in R
    tol = 1e-10
    independent_indices = []
    for i in range(min(m, n)):
        if abs(R[i, i]) > tol:
            independent_indices.append(i)
    
    return independent_indices

# Example
vectors = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([1, 1, 0]),  # This is dependent (v1 + v2)
    np.array([0, 0, 1]),
]

basis_indices = find_basis(vectors)
print("Original vectors:")
for i, v in enumerate(vectors):
    print(f"  v{i+1}: {v}")

print(f"\nBasis indices: {basis_indices}")
print("Basis vectors:")
for i in basis_indices:
    print(f"  v{i+1}: {vectors[i]}")


# ============================================
# 4. MATRIX RANK
# ============================================

print("\n" + "=" * 50)
print("4. MATRIX RANK")
print("=" * 50)

# Full rank matrix
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 10]  # Changed from 9 to make full rank
])
print(f"Matrix A:\n{A}")
print(f"Rank of A: {np.linalg.matrix_rank(A)}")
print(f"Shape: {A.shape}")

# Rank-deficient matrix
B = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]  # This is 2*row1 + row2 approximately
])
print(f"\nMatrix B:\n{B}")
print(f"Rank of B: {np.linalg.matrix_rank(B)}")
print("(Row 3 is a linear combination of rows 1 and 2)")

# Rank and solvability of linear systems
print("\n--- Rank and Linear Systems ---")
# Ax = b has a solution iff rank([A|b]) = rank(A)


# ============================================
# 5. CHANGE OF BASIS
# ============================================

print("\n" + "=" * 50)
print("5. CHANGE OF BASIS")
print("=" * 50)

def change_of_basis(v, old_basis, new_basis):
    """
    Express vector v (in standard basis) in terms of new_basis.
    """
    # First, express v in terms of old_basis
    # Then transform to new_basis
    
    # Create transformation matrix: columns are new basis vectors
    P = np.column_stack(new_basis)
    
    # Solve P @ coords = v for coords
    coords = np.linalg.solve(P, v)
    return coords

# Standard basis
e1 = np.array([1, 0])
e2 = np.array([0, 1])
standard_basis = [e1, e2]

# New basis (45 degree rotation)
b1 = np.array([1, 1]) / np.sqrt(2)
b2 = np.array([-1, 1]) / np.sqrt(2)
new_basis = [b1, b2]

# Vector in standard coordinates
v = np.array([3, 1])

# Express in new basis
new_coords = change_of_basis(v, standard_basis, new_basis)
print(f"Vector v in standard basis: {v}")
print(f"New basis vectors:")
print(f"  b1: {b1}")
print(f"  b2: {b2}")
print(f"Vector v in new basis: {new_coords}")

# Verify
reconstructed = new_coords[0] * b1 + new_coords[1] * b2
print(f"Reconstructed: {reconstructed}")
print(f"Match: {np.allclose(v, reconstructed)}")


# ============================================
# 6. COLUMN SPACE AND NULL SPACE
# ============================================

print("\n" + "=" * 50)
print("6. COLUMN SPACE AND NULL SPACE")
print("=" * 50)

A = np.array([
    [1, 2, 1],
    [2, 4, 3],
    [3, 6, 4]
])

print(f"Matrix A:\n{A}")

# Column space: span of columns
# Use SVD to find basis for column space
U, S, Vt = np.linalg.svd(A)
rank = np.sum(S > 1e-10)
column_space_basis = U[:, :rank]
print(f"\nRank: {rank}")
print(f"Basis for column space (orthonormal):\n{column_space_basis}")

# Null space: vectors x where Ax = 0
# Null space is spanned by columns of V corresponding to zero singular values
null_space_basis = Vt[rank:, :].T
if null_space_basis.size > 0:
    print(f"\nBasis for null space:\n{null_space_basis}")
    # Verify
    print(f"A @ null_vector:\n{A @ null_space_basis}")


# ============================================
# 7. VISUALIZATION
# ============================================

print("\n" + "=" * 50)
print("7. VISUALIZATION")
print("=" * 50)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Span of one vector (a line)
ax1 = axes[0]
v = np.array([2, 1])
t = np.linspace(-2, 2, 100)
line = np.outer(t, v)
ax1.plot(line[:, 0], line[:, 1], 'b-', linewidth=2, label='span({v})')
ax1.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='v')
ax1.set_xlim(-5, 5)
ax1.set_ylim(-3, 3)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.legend()
ax1.set_title('Span of 1 Vector = Line')

# Plot 2: Independent vs Dependent vectors
ax2 = axes[1]
# Independent
v1 = np.array([2, 1])
v2 = np.array([1, 2])
ax2.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='g', label='v1 (ind)')
ax2.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='g')
# Dependent
v3 = np.array([-2, -1])
v4 = np.array([-4, -2])  # = 2*v3
ax2.quiver(0, 0, v3[0], v3[1], angles='xy', scale_units='xy', scale=1, color='r', label='v3 (dep)')
ax2.quiver(0, 0, v4[0], v4[1], angles='xy', scale_units='xy', scale=1, color='r', alpha=0.5)
ax2.set_xlim(-5, 3)
ax2.set_ylim(-3, 3)
ax2.set_aspect('equal')
ax2.grid(True)
ax2.legend()
ax2.set_title('Independent (green) vs Dependent (red)')

# Plot 3: Different bases representing same vector
ax3 = axes[2]
v = np.array([3, 2])
# Standard basis
e1 = np.array([1, 0])
e2 = np.array([0, 1])
# New basis
b1 = np.array([1, 0.5])
b2 = np.array([0.5, 1])

ax3.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='purple', 
           linewidth=3, label='v = [3,2]')
ax3.quiver(0, 0, e1[0]*3, e1[1]*3, angles='xy', scale_units='xy', scale=1, color='b', alpha=0.5)
ax3.quiver(e1[0]*3, e1[1]*3, e2[0]*2, e2[1]*2, angles='xy', scale_units='xy', scale=1, color='b', 
           alpha=0.5, label='Standard: 3e1 + 2e2')

# Find coordinates in new basis
P = np.column_stack([b1, b2])
coords = np.linalg.solve(P, v)
ax3.quiver(0, 0, b1[0]*coords[0], b1[1]*coords[0], angles='xy', scale_units='xy', scale=1, 
           color='orange', alpha=0.7)
ax3.quiver(b1[0]*coords[0], b1[1]*coords[0], b2[0]*coords[1], b2[1]*coords[1], 
           angles='xy', scale_units='xy', scale=1, color='orange', alpha=0.7,
           label=f'New: {coords[0]:.1f}b1 + {coords[1]:.1f}b2')

ax3.set_xlim(-0.5, 4)
ax3.set_ylim(-0.5, 3)
ax3.set_aspect('equal')
ax3.grid(True)
ax3.legend()
ax3.set_title('Same Vector, Different Bases')

plt.tight_layout()
plt.savefig('span_basis_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'span_basis_visualization.png'")


# ============================================
# 8. ML APPLICATION: Feature Selection
# ============================================

print("\n" + "=" * 50)
print("8. ML APPLICATION: Feature Selection")
print("=" * 50)

# Simulated feature matrix with some redundant features
np.random.seed(42)

# Create original independent features
X_independent = np.random.randn(100, 3)

# Create redundant features (linear combinations)
X_redundant = np.column_stack([
    X_independent,
    X_independent[:, 0] + X_independent[:, 1],  # Feature 4 = F1 + F2
    2 * X_independent[:, 2],  # Feature 5 = 2 * F3
])

print(f"Original feature matrix shape: {X_redundant.shape}")
print(f"Matrix rank: {np.linalg.matrix_rank(X_redundant)}")
print("(Rank < number of features indicates redundancy)")

# Identify independent features using SVD
U, S, Vt = np.linalg.svd(X_redundant, full_matrices=False)
print(f"\nSingular values: {S}")

# Significant singular values indicate important dimensions
threshold = 1e-10
n_components = np.sum(S > threshold * S[0])
print(f"Number of significant components: {n_components}")

# In practice, you'd keep features corresponding to top singular values
print("\nIn ML: Remove linearly dependent features to avoid multicollinearity!")


print("\n" + "=" * 50)
print("Day 2 Complete! 🎉")
print("=" * 50)
