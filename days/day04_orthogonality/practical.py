"""
Day 4: Orthogonality & Orthogonal Matrices
Practical Implementation in Python

Author: Amey Prakash Sawant
100 Days of AI/ML Journey
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. ORTHOGONALITY CHECK
# ============================================

print("=" * 50)
print("1. ORTHOGONALITY CHECK")
print("=" * 50)

def are_orthogonal(u, v, tol=1e-10):
    """Check if two vectors are orthogonal"""
    return abs(np.dot(u, v)) < tol

# Example: Standard basis vectors
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

print(f"e1 = {e1}")
print(f"e2 = {e2}")
print(f"e1 · e2 = {np.dot(e1, e2)}")
print(f"Are e1 and e2 orthogonal? {are_orthogonal(e1, e2)}")

# Non-orthogonal vectors
u = np.array([1, 1, 0])
v = np.array([1, 0, 0])
print(f"\nu = {u}")
print(f"v = {v}")
print(f"u · v = {np.dot(u, v)}")
print(f"Are u and v orthogonal? {are_orthogonal(u, v)}")


# ============================================
# 2. ORTHONORMAL VECTORS
# ============================================

print("\n" + "=" * 50)
print("2. ORTHONORMAL VECTORS")
print("=" * 50)

def is_orthonormal(vectors, tol=1e-10):
    """Check if a set of vectors is orthonormal"""
    n = len(vectors)
    for i in range(n):
        # Check unit length
        if abs(np.linalg.norm(vectors[i]) - 1) > tol:
            return False, f"Vector {i} is not unit length"
        # Check orthogonality with others
        for j in range(i+1, n):
            if abs(np.dot(vectors[i], vectors[j])) > tol:
                return False, f"Vectors {i} and {j} are not orthogonal"
    return True, "Vectors are orthonormal"

# Check standard basis
basis = [e1, e2, e3]
result, msg = is_orthonormal(basis)
print(f"Standard basis orthonormal? {result} - {msg}")

# Create orthonormal basis (not standard)
u1 = np.array([1, 1, 0]) / np.sqrt(2)
u2 = np.array([-1, 1, 0]) / np.sqrt(2)
u3 = np.array([0, 0, 1])

custom_basis = [u1, u2, u3]
result, msg = is_orthonormal(custom_basis)
print(f"\nCustom basis: {[v.round(3) for v in custom_basis]}")
print(f"Orthonormal? {result} - {msg}")


# ============================================
# 3. ORTHOGONAL MATRICES
# ============================================

print("\n" + "=" * 50)
print("3. ORTHOGONAL MATRICES")
print("=" * 50)

def is_orthogonal_matrix(Q, tol=1e-10):
    """Check if a matrix is orthogonal"""
    n = Q.shape[0]
    QTQ = Q.T @ Q
    return np.allclose(QTQ, np.eye(n), atol=tol)

# Create orthogonal matrix from orthonormal vectors
Q = np.column_stack([u1, u2, u3])
print(f"Matrix Q:\n{Q}")
print(f"\nQ^T @ Q:\n{(Q.T @ Q).round(10)}")
print(f"\nIs Q orthogonal? {is_orthogonal_matrix(Q)}")
print(f"\nQ^-1 (computed):\n{np.linalg.inv(Q).round(10)}")
print(f"\nQ^T (should equal Q^-1):\n{Q.T.round(10)}")

# Key property: inverse = transpose
print(f"\nQ^-1 == Q^T? {np.allclose(np.linalg.inv(Q), Q.T)}")


# ============================================
# 4. ROTATION MATRICES
# ============================================

print("\n" + "=" * 50)
print("4. ROTATION MATRICES")
print("=" * 50)

def rotation_matrix_2d(theta):
    """Create 2D rotation matrix for angle theta (in radians)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s, c]])

def rotation_matrix_3d_z(theta):
    """Rotation around z-axis"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

# 90 degree rotation
theta = np.pi / 2  # 90 degrees
R = rotation_matrix_2d(theta)
print(f"90° rotation matrix:\n{R.round(10)}")

# Rotate a vector
v = np.array([1, 0])
v_rotated = R @ v
print(f"\nOriginal vector: {v}")
print(f"Rotated vector: {v_rotated.round(10)}")

# Check orthogonality
print(f"\nIs R orthogonal? {is_orthogonal_matrix(R)}")
print(f"det(R) = {np.linalg.det(R):.4f} (1 for rotation, -1 for reflection)")

# Composing rotations
R1 = rotation_matrix_2d(np.pi/4)  # 45°
R2 = rotation_matrix_2d(np.pi/4)  # 45°
R_composed = R1 @ R2  # Should be 90°
R_90 = rotation_matrix_2d(np.pi/2)
print(f"\nR(45°) @ R(45°) == R(90°)? {np.allclose(R_composed, R_90)}")


# ============================================
# 5. GRAM-SCHMIDT ORTHOGONALIZATION
# ============================================

print("\n" + "=" * 50)
print("5. GRAM-SCHMIDT ORTHOGONALIZATION")
print("=" * 50)

def gram_schmidt(vectors):
    """
    Perform Gram-Schmidt orthogonalization.
    Returns orthonormal vectors.
    """
    orthonormal = []
    
    for v in vectors:
        # Subtract projections onto all previous orthonormal vectors
        u = v.copy().astype(float)
        for prev in orthonormal:
            u = u - np.dot(v, prev) * prev
        
        # Normalize
        norm = np.linalg.norm(u)
        if norm > 1e-10:  # Avoid division by zero
            orthonormal.append(u / norm)
    
    return orthonormal

# Example: Arbitrary vectors
v1 = np.array([1, 1, 0])
v2 = np.array([1, 0, 1])
v3 = np.array([0, 1, 1])

print("Original vectors:")
print(f"  v1 = {v1}")
print(f"  v2 = {v2}")
print(f"  v3 = {v3}")

orthonormal_vecs = gram_schmidt([v1, v2, v3])
print("\nOrthonormalized vectors:")
for i, u in enumerate(orthonormal_vecs):
    print(f"  u{i+1} = {u.round(4)}")

# Verify orthonormality
result, msg = is_orthonormal(orthonormal_vecs)
print(f"\nOrthonormal? {result} - {msg}")


# ============================================
# 6. QR DECOMPOSITION
# ============================================

print("\n" + "=" * 50)
print("6. QR DECOMPOSITION")
print("=" * 50)

A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

print(f"Matrix A:\n{A}")

# QR decomposition using NumPy
Q, R = np.linalg.qr(A)
print(f"\nQ (orthogonal):\n{Q.round(4)}")
print(f"\nR (upper triangular):\n{R.round(4)}")

# Verify
print(f"\nQ @ R:\n{(Q @ R).round(4)}")
print(f"Q @ R == A? {np.allclose(Q @ R, A)}")

# Q is orthogonal
print(f"\nIs Q orthogonal? {is_orthogonal_matrix(Q)}")


# ============================================
# 7. ORTHOGONAL PROJECTION
# ============================================

print("\n" + "=" * 50)
print("7. ORTHOGONAL PROJECTION")
print("=" * 50)

def project_onto_subspace(b, A):
    """
    Project vector b onto column space of A.
    Returns projection and projection matrix.
    """
    # P = A(A^T A)^-1 A^T
    ATA = A.T @ A
    ATA_inv = np.linalg.inv(ATA)
    P = A @ ATA_inv @ A.T
    projection = P @ b
    return projection, P

# Example: Project onto a plane
# Plane spanned by [1,0,0] and [0,1,0]
A = np.array([[1, 0],
              [0, 1],
              [0, 0]], dtype=float)

b = np.array([1, 2, 3])  # Vector to project

proj, P = project_onto_subspace(b, A)
print(f"Vector b: {b}")
print(f"Subspace (xy-plane) basis:\n{A}")
print(f"\nProjection of b: {proj}")
print(f"(z-component removed, as expected)")

# Verify projection matrix properties
print(f"\nProjection matrix P:\n{P.round(4)}")
print(f"P^2 = P? {np.allclose(P @ P, P)}")
print(f"P^T = P? {np.allclose(P.T, P)}")


# ============================================
# 8. PRESERVING LENGTHS AND ANGLES
# ============================================

print("\n" + "=" * 50)
print("8. PRESERVING LENGTHS AND ANGLES")
print("=" * 50)

Q = rotation_matrix_2d(np.pi/3)  # 60 degree rotation
x = np.array([3, 4])
y = np.array([1, 2])

Qx = Q @ x
Qy = Q @ y

print(f"Original x: {x}, ||x|| = {np.linalg.norm(x):.4f}")
print(f"Rotated Qx: {Qx.round(4)}, ||Qx|| = {np.linalg.norm(Qx):.4f}")
print(f"\nLengths preserved? {np.isclose(np.linalg.norm(x), np.linalg.norm(Qx))}")

# Check angles (via dot product)
original_dot = np.dot(x, y)
rotated_dot = np.dot(Qx, Qy)
print(f"\nOriginal x·y = {original_dot:.4f}")
print(f"Rotated (Qx)·(Qy) = {rotated_dot:.4f}")
print(f"Dot product preserved? {np.isclose(original_dot, rotated_dot)}")


# ============================================
# 9. VISUALIZATION
# ============================================

print("\n" + "=" * 50)
print("9. VISUALIZATION")
print("=" * 50)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: 2D Rotation
ax1 = axes[0]
angles = [0, 30, 60, 90, 120, 150, 180]
v = np.array([1, 0])
colors = plt.cm.rainbow(np.linspace(0, 1, len(angles)))

for angle, color in zip(angles, colors):
    R = rotation_matrix_2d(np.radians(angle))
    rotated = R @ v
    ax1.quiver(0, 0, rotated[0], rotated[1], angles='xy', scale_units='xy', 
               scale=1, color=color, label=f'{angle}°')

ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.legend(loc='upper right')
ax1.set_title('2D Rotation')

# Plot 2: Gram-Schmidt Process
ax2 = axes[1]
v1_orig = np.array([2, 1])
v2_orig = np.array([1, 2])

# Original vectors
ax2.quiver(0, 0, v1_orig[0], v1_orig[1], angles='xy', scale_units='xy', 
           scale=1, color='blue', alpha=0.5, label='v1 (original)')
ax2.quiver(0, 0, v2_orig[0], v2_orig[1], angles='xy', scale_units='xy', 
           scale=1, color='red', alpha=0.5, label='v2 (original)')

# Orthonormalized
ortho = gram_schmidt([v1_orig, v2_orig])
ax2.quiver(0, 0, ortho[0][0], ortho[0][1], angles='xy', scale_units='xy', 
           scale=1, color='blue', linewidth=2, label='u1 (orthonormal)')
ax2.quiver(0, 0, ortho[1][0], ortho[1][1], angles='xy', scale_units='xy', 
           scale=1, color='red', linewidth=2, label='u2 (orthonormal)')

ax2.set_xlim(-0.5, 2.5)
ax2.set_ylim(-0.5, 2.5)
ax2.set_aspect('equal')
ax2.grid(True)
ax2.legend()
ax2.set_title('Gram-Schmidt Orthogonalization')

# Plot 3: Orthogonal Projection
ax3 = axes[2]
# Project onto line y = x/2
line_vec = np.array([2, 1])
line_vec = line_vec / np.linalg.norm(line_vec)  # Unit vector

# Points to project
points = np.array([[1, 3], [2, 2], [3, 1], [0, 2]])
projections = np.array([np.dot(p, line_vec) * line_vec for p in points])

# Draw line
t = np.linspace(-1, 4, 100)
ax3.plot(t * line_vec[0], t * line_vec[1], 'g-', linewidth=2, label='Subspace')

# Draw points and projections
for p, proj in zip(points, projections):
    ax3.plot(*p, 'bo', markersize=8)
    ax3.plot(*proj, 'ro', markersize=8)
    ax3.plot([p[0], proj[0]], [p[1], proj[1]], 'k--', alpha=0.5)

ax3.set_xlim(-1, 4)
ax3.set_ylim(-1, 4)
ax3.set_aspect('equal')
ax3.grid(True)
ax3.legend(['Subspace', 'Original', 'Projection'])
ax3.set_title('Orthogonal Projection onto Line')

plt.tight_layout()
plt.savefig('orthogonality_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'orthogonality_visualization.png'")


print("\n" + "=" * 50)
print("Day 4 Complete! 🎉")
print("=" * 50)
