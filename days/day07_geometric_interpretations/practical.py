# Day 7: Seeing Linear Algebra
# What matrices actually do to shapes and spaces
# Amey Prakash Sawant

print("Day 7: Geometric Interpretations")
print("=" * 32)

# What matrices do to shapes - let's see it!

# Simple 2x2 matrix transformations
print("\nTransformations - how matrices change shapes:")

# Scaling matrix - makes things bigger/smaller
scaling = [[2, 0],
           [0, 3]]  # x gets 2x bigger, y gets 3x bigger

print(f"Scaling matrix: {scaling}")
print("Effect: stretches shapes")

# Rotation matrix - spins things around
import math
angle = math.pi/4  # 45 degrees
rotation = [[math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)]]

print(f"\nRotation matrix (45°): [[{rotation[0][0]:.2f}, {rotation[0][1]:.2f}],")
print(f"                       [{rotation[1][0]:.2f}, {rotation[1][1]:.2f}]]")
print("Effect: rotates shapes")

# Reflection matrix - flips things
reflection = [[1, 0],
              [0, -1]]  # flip over x-axis

print(f"\nReflection matrix: {reflection}")
print("Effect: flips shapes upside down")

# Let's transform a simple square
square_points = [[0, 1, 1, 0],  # x coordinates
                 [0, 0, 1, 1]]  # y coordinates

print("\nTransforming a square:")
print("Original square corners:") 
for i in range(4):
    print(f"  ({square_points[0][i]}, {square_points[1][i]})")
# Let's see what scaling does
def matrix_multiply_point(matrix, point):
    # Manual matrix multiplication for a point
    x = matrix[0][0] * point[0] + matrix[0][1] * point[1]
    y = matrix[1][0] * point[0] + matrix[1][1] * point[1]
    return [x, y]

print("\nAfter scaling (2x in x, 3x in y):")
for i in range(4):
    original = [square_points[0][i], square_points[1][i]]
    scaled = matrix_multiply_point(scaling, original)
    print(f"  ({original[0]}, {original[1]}) → ({scaled[0]}, {scaled[1]})")

print("\nThe square became a rectangle!")

# Determinant tells us about area change
def determinant_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

scale_det = determinant_2x2(scaling)
print(f"\nScaling determinant = {scale_det}")
print(f"Area changed by factor of {scale_det} (was 1, now {scale_det})")

reflect_det = determinant_2x2(reflection)
print(f"\nReflection determinant = {reflect_det}")
print("Negative = flipped orientation!")

print("\nKey insights:")
print("- Matrices transform shapes")
print("- Determinant = area scaling factor")
print("- Negative det = flipped orientation")
print("- Zero det = squashed to line (no area)")

print("\nDay 7 complete! ✅")

# Import numpy and matplotlib for visualizations
import numpy as np
import matplotlib.pyplot as plt

def plot_transformation_2d(A, title, ax):
    """Plot how a 2D transformation affects basis vectors"""
    # Standard basis vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    # Transformed basis vectors
    ae1 = A @ e1
    ae2 = A @ e2
    
    ax.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', width=0.05, label='Original basis')
    ax.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', width=0.05)
    ax.quiver(0, 0, ae1[0], ae1[1], angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.05, label='Transformed basis')
    ax.quiver(0, 0, ae2[0], ae2[1], angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.05)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\ndet = {np.linalg.det(A):.2f}')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

# Different transformations
transformations = {
    'Identity': np.eye(2),
    'Scale (2, 0.5)': np.diag([2, 0.5]),
    'Rotation (45°)': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                 [np.sin(np.pi/4), np.cos(np.pi/4)]]),
    'Shear': np.array([[1, 1], [0, 1]]),
    'Reflection (y)': np.array([[-1, 0], [0, 1]]),
    'Projection (x-axis)': np.array([[1, 0], [0, 0]])
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (name, A) in zip(axes.flatten(), transformations.items()):
    plot_transformation_2d(A, name, ax)

plt.tight_layout()
plt.savefig('transformations_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Transformations saved as 'transformations_visualization.png'")


# ============================================
# 2. EIGENVECTORS VISUALIZATION
# ============================================

print("\n" + "=" * 50)
print("2. EIGENVECTORS - SPECIAL DIRECTIONS")
print("=" * 50)

A = np.array([[3, 1],
              [0, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Matrix A:\n{A}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    Av = A @ v
    lambda_v = lam * v
    print(f"\nEigenvector {i+1}: {v}")
    print(f"A @ v = {Av}")
    print(f"λ @ v = {lambda_v}")
    print(f"Match: {np.allclose(Av, lambda_v)}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Grid of random vectors
np.random.seed(42)
n_vectors = 30
angles = np.linspace(0, 2*np.pi, n_vectors)
vectors = np.array([[np.cos(a), np.sin(a)] for a in angles])

# Plot original and transformed
for v in vectors:
    Av = A @ v
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', alpha=0.3, width=0.01)
    ax.quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1, 
              color='red', alpha=0.3, width=0.01)

# Plot eigenvectors (they only scale, don't rotate!)
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    vec = vec / np.linalg.norm(vec)  # Normalize
    Avec = A @ vec
    ax.quiver(0, 0, vec[0]*2, vec[1]*2, angles='xy', scale_units='xy', scale=1, 
              color='green', width=0.03, label=f'Eigenvector (λ={val:.1f})' if i == 0 else '')
    ax.quiver(0, 0, Avec[0], Avec[1], angles='xy', scale_units='xy', scale=1, 
              color='darkgreen', width=0.03)

ax.set_xlim(-5, 5)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.set_title('Eigenvectors: Only Scale, Never Rotate\n(Blue=original, Red=transformed, Green=eigenvectors)')
ax.grid(True, alpha=0.3)
ax.legend()

plt.savefig('eigenvectors_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Eigenvectors saved as 'eigenvectors_visualization.png'")


# ============================================
# 3. SVD GEOMETRY - UNIT CIRCLE TO ELLIPSE
# ============================================

print("\n" + "=" * 50)
print("3. SVD GEOMETRY")
print("=" * 50)

A = np.array([[2, 1],
              [1, 2]])

U, S, Vt = np.linalg.svd(A)

print(f"Matrix A:\n{A}")
print(f"Singular values: {S}")
print(f"U:\n{U}")
print(f"V^T:\n{Vt}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Unit circle
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# Step 1: Original circle
axes[0].plot(circle[0], circle[1], 'b-')
axes[0].quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='red', width=0.03)
axes[0].quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='green', width=0.03)
axes[0].set_title('Step 0: Unit Circle')
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-3, 3)
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Step 2: After V^T (rotation in input space)
step1 = Vt @ circle
axes[1].plot(step1[0], step1[1], 'b-')
axes[1].quiver(0, 0, Vt[0, 0], Vt[1, 0], angles='xy', scale_units='xy', scale=1, color='red', width=0.03)
axes[1].quiver(0, 0, Vt[0, 1], Vt[1, 1], angles='xy', scale_units='xy', scale=1, color='green', width=0.03)
axes[1].set_title('Step 1: V^T (Rotate)')
axes[1].set_xlim(-3, 3)
axes[1].set_ylim(-3, 3)
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

# Step 3: After Σ (scaling)
Sigma = np.diag(S)
step2 = Sigma @ step1
axes[2].plot(step2[0], step2[1], 'b-')
axes[2].quiver(0, 0, S[0]*Vt[0, 0], S[0]*Vt[1, 0], angles='xy', scale_units='xy', scale=1, color='red', width=0.03)
axes[2].quiver(0, 0, S[1]*Vt[0, 1], S[1]*Vt[1, 1], angles='xy', scale_units='xy', scale=1, color='green', width=0.03)
axes[2].set_title(f'Step 2: Σ (Scale σ₁={S[0]:.2f}, σ₂={S[1]:.2f})')
axes[2].set_xlim(-4, 4)
axes[2].set_ylim(-4, 4)
axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.3)

# Step 4: After U (rotation in output space)
step3 = U @ step2
axes[3].plot(step3[0], step3[1], 'b-')
# Plot final axes (columns of U scaled by singular values)
axes[3].quiver(0, 0, S[0]*U[0, 0], S[0]*U[1, 0], angles='xy', scale_units='xy', scale=1, color='red', width=0.03, label='σ₁u₁')
axes[3].quiver(0, 0, S[1]*U[0, 1], S[1]*U[1, 1], angles='xy', scale_units='xy', scale=1, color='green', width=0.03, label='σ₂u₂')
axes[3].set_title('Step 3: U (Rotate) = Final Ellipse')
axes[3].set_xlim(-4, 4)
axes[3].set_ylim(-4, 4)
axes[3].set_aspect('equal')
axes[3].grid(True, alpha=0.3)
axes[3].legend()

plt.tight_layout()
plt.savefig('svd_geometry_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("SVD geometry saved as 'svd_geometry_visualization.png'")


# ============================================
# 4. PROJECTION GEOMETRY
# ============================================

print("\n" + "=" * 50)
print("4. PROJECTION GEOMETRY")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Projection onto a line
ax1 = axes[0]
a = np.array([2, 1])  # Line direction
b = np.array([1, 2])  # Vector to project

# Projection
proj = (np.dot(a, b) / np.dot(a, a)) * a
residual = b - proj

# Plot
t = np.linspace(-1, 3, 100)
line = np.outer(t, a / np.linalg.norm(a))
ax1.plot(line[:, 0], line[:, 1], 'g-', linewidth=2, label='Subspace (line)')
ax1.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.03, label='b (original)')
ax1.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.03, label='proj(b)')
ax1.quiver(proj[0], proj[1], residual[0], residual[1], angles='xy', scale_units='xy', scale=1, color='purple', width=0.02, label='residual (⊥)')
ax1.plot([b[0], proj[0]], [b[1], proj[1]], 'k--', alpha=0.5)
ax1.set_xlim(-1, 3)
ax1.set_ylim(-1, 3)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_title('Projection onto Line')

# Projection onto a plane (in 3D, shown as 2D slice)
ax2 = axes[1]
# Two vectors spanning a plane
a1 = np.array([1, 0])
a2 = np.array([0, 1])
A = np.column_stack([a1, a2])

# Vector to project (pretend we're in 3D but showing 2D projection)
# Show multiple points being projected
np.random.seed(42)
points = np.random.randn(10, 2) * 2

for p in points:
    # Already in the plane, just plot
    ax2.plot(p[0], p[1], 'bo', markersize=8)

# Show column space shaded
x_range = np.linspace(-3, 3, 2)
y_range = np.linspace(-3, 3, 2)
ax2.fill([x_range[0], x_range[1], x_range[1], x_range[0]], 
         [y_range[0], y_range[0], y_range[1], y_range[1]], 
         alpha=0.2, color='green', label='Column space')

ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_title('Projection onto Plane (2D view)')

plt.tight_layout()
plt.savefig('projection_geometry_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Projection geometry saved as 'projection_geometry_visualization.png'")


# ============================================
# 5. LEAST SQUARES GEOMETRY
# ============================================

print("\n" + "=" * 50)
print("5. LEAST SQUARES GEOMETRY")
print("=" * 50)

# Create an overdetermined system
A = np.array([[1, 1],
              [1, 2],
              [1, 3]])
b = np.array([1, 2, 2])

# Solve least squares
x_ls, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
b_projected = A @ x_ls
residual = b - b_projected

print(f"Matrix A:\n{A}")
print(f"Vector b: {b}")
print(f"Least squares solution x: {x_ls}")
print(f"Projected b (Ax): {b_projected}")
print(f"Residual: {residual}")
print(f"||residual||: {np.linalg.norm(residual):.4f}")

# Verify residual is orthogonal to column space
print(f"\nA^T @ residual (should be ~0): {A.T @ residual}")


# ============================================
# 6. EIGENVALUE POWERS AND STABILITY
# ============================================

print("\n" + "=" * 50)
print("6. EIGENVALUE POWERS AND STABILITY")
print("=" * 50)

def analyze_matrix_power(A, name):
    """Analyze behavior of A^n as n grows"""
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = np.max(np.abs(eigenvalues))
    
    print(f"\n{name}:")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Spectral radius: {spectral_radius:.4f}")
    
    if spectral_radius < 1:
        print("→ A^n → 0 as n → ∞ (stable/convergent)")
    elif spectral_radius > 1:
        print("→ A^n → ∞ as n → ∞ (unstable/divergent)")
    else:
        print("→ A^n bounded but may not converge")
    
    return eigenvalues, spectral_radius

# Different matrices
matrices = {
    'Stable (eigenvalues < 1)': np.array([[0.5, 0.1], [0.1, 0.5]]),
    'Unstable (eigenvalues > 1)': np.array([[1.1, 0.1], [0.1, 1.2]]),
    'Marginally stable': np.array([[0, -1], [1, 0]])  # Rotation
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, A) in zip(axes, matrices.items()):
    eigenvalues, sr = analyze_matrix_power(A, name)
    
    # Track norm of A^n
    ns = range(0, 50)
    norms = []
    An = np.eye(2)
    for n in ns:
        norms.append(np.linalg.norm(An, 'fro'))
        An = An @ A
    
    ax.semilogy(ns, norms)
    ax.set_xlabel('n')
    ax.set_ylabel('||A^n||')
    ax.set_title(f'{name}\nρ(A) = {sr:.2f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('matrix_stability_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nStability saved as 'matrix_stability_visualization.png'")


# ============================================
# 7. COLUMN SPACE AND NULL SPACE
# ============================================

print("\n" + "=" * 50)
print("7. COLUMN SPACE AND NULL SPACE")
print("=" * 50)

# Rank-deficient matrix
A = np.array([[1, 2, 3],
              [2, 4, 6]])  # Row 2 = 2 * Row 1

print(f"Matrix A:\n{A}")
print(f"Rank: {np.linalg.matrix_rank(A)}")

# Column space (via SVD)
U, S, Vt = np.linalg.svd(A)
rank = np.sum(S > 1e-10)
col_space = U[:, :rank]
null_space = Vt[rank:, :].T

print(f"\nColumn space basis:\n{col_space}")
print(f"Null space basis:\n{null_space}")

# Verify null space
for i in range(null_space.shape[1]):
    v = null_space[:, i]
    print(f"\nNull space vector {i+1}: {v}")
    print(f"A @ v: {A @ v}")


print("\n" + "=" * 50)
print("Day 7 Complete! 🎉")
print("=" * 50)
print("\nLinear Algebra Week Complete! 🎓")
