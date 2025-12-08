# Day 2: Span, Linear Independence & Basis
# Understanding how vectors work together
# Amey Prakash Sawant

import numpy as np
import matplotlib.pyplot as plt

# Linear combinations - mixing vectors
print("Day 2: Vector Combinations and Independence")
print("=" * 40)

# Basic vectors
v1 = [1, 0]
v2 = [0, 1]

# Make new vectors by mixing these
def combine_vectors(vectors, amounts):
    result = [0, 0]
    for i, vector in enumerate(vectors):
        amount = amounts[i]
        for j in range(len(vector)):
            result[j] += amount * vector[j]
    return result

# Example: make [3, 4] using standard basis
target = [3, 4]
amounts = [3, 4]  # 3*v1 + 4*v2
result = combine_vectors([v1, v2], amounts)
print(f"Basic vectors: {v1}, {v2}")
print(f"Mix with amounts {amounts}: {result}")
print(f"Target was {target} - match!")

# Try different basis vectors
new_v1 = [1, 1]
new_v2 = [1, -1]
# To make [3, 4] with these vectors:
# a*[1,1] + b*[1,-1] = [3,4]
# a + b = 3, a - b = 4
# Solving: a = 3.5, b = -0.5
new_amounts = [3.5, -0.5]
new_result = combine_vectors([new_v1, new_v2], new_amounts)
print(f"\nDifferent basis: {new_v1}, {new_v2}")
print(f"Mix with amounts {new_amounts}: {new_result}")
print(f"Same target {target}!")

# Linear independence - can one vector be made from others?
print("\nLinear Independence:")
print("-" * 20)

# Independent vectors - can't make one from the other
vec1 = [1, 0]
vec2 = [0, 1]
print(f"Vectors {vec1} and {vec2}:")
print("Can we make [1,0] from some amount of [0,1]? No!")
print("These are INDEPENDENT")

# Dependent vectors - one is just a multiple of the other
vec3 = [1, 2]
vec4 = [2, 4]  # This is just 2 * [1,2]
print(f"\nVectors {vec3} and {vec4}:")
print(f"{vec4} = 2 * {vec3}")
print("These are DEPENDENT")

# Three 2D vectors - always dependent
vec5 = [1, 0]
vec6 = [0, 1]
vec7 = [1, 1]
print(f"\nThree 2D vectors: {vec5}, {vec6}, {vec7}")
# [1,1] = 1*[1,0] + 1*[0,1]
print("[1,1] = 1*[1,0] + 1*[0,1]")
print("In 2D space, max 2 independent vectors")

# Check if vectors are dependent (simple version)
def vectors_dependent_2d(v1, v2):
    # If v1 = k*v2 for some k, they're dependent
    # Check if v1[0]/v2[0] == v1[1]/v2[1] (avoiding division by zero)
    if v2[0] == 0 and v2[1] == 0:
        return True
    if v2[0] == 0:
        return v1[0] == 0
    if v2[1] == 0:
        return v1[1] == 0
    
    ratio1 = v1[0] / v2[0] if v2[0] != 0 else float('inf')
    ratio2 = v1[1] / v2[1] if v2[1] != 0 else float('inf')
    
    return abs(ratio1 - ratio2) < 0.0001

test1 = [2, 4]
test2 = [1, 2]
print(f"\nAre {test1} and {test2} dependent? {vectors_dependent_2d(test1, test2)}")

test3 = [1, 0]
test4 = [0, 1]
print(f"Are {test3} and {test4} dependent? {vectors_dependent_2d(test3, test4)}")

# Basis vectors - minimal set that spans the space
print("\nBasis Vectors:")
print("-" * 13)

print("Standard 2D basis: [1,0] and [0,1]")
print("Can make any 2D vector: a*[1,0] + b*[0,1] = [a,b]")

print("\nAlternative 2D basis: [1,1] and [1,-1]")
print("Still spans all 2D space, just different coordinates")

# Finding which vectors form a basis (simple approach)
def find_basis_2d(vectors):
    basis = []
    for vec in vectors:
        # Check if this vector is independent from what we have
        if len(basis) == 0:
            basis.append(vec)
        elif len(basis) == 1:
            if not vectors_dependent_2d(vec, basis[0]):
                basis.append(vec)
        else:
            break  # Already have 2 vectors in 2D
    return basis

test_vectors = [[2, 4], [1, 0], [0, 1], [3, 6]]
basis = find_basis_2d(test_vectors)
print(f"\nFrom vectors {test_vectors}")
print(f"Basis found: {basis}")

print("\nDay 2 complete! ✅")
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
