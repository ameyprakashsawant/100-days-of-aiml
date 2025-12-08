# Day 4: Orthogonality & Perpendicular Vectors
# When vectors are perpendicular to each other  
# Amey Prakash Sawant

import math

# Orthogonal = perpendicular (90 degree angle)
print("Day 4: Orthogonal Vectors")
print("=" * 25)

# Check if vectors are perpendicular
def are_perpendicular(vec1, vec2):
    # Vectors are perpendicular if their dot product = 0
    dot_product = sum(vec1[i] * vec2[i] for i in range(len(vec1)))
    return abs(dot_product) < 0.0001  # Close to zero

# Standard basis vectors are perpendicular
e1 = [1, 0, 0]  # x-axis
e2 = [0, 1, 0]  # y-axis  
e3 = [0, 0, 1]  # z-axis

print(f"Basis vectors:")
print(f"e1 = {e1}")
print(f"e2 = {e2}")
print(f"e3 = {e3}")

dot_e1_e2 = sum(e1[i] * e2[i] for i in range(len(e1)))
print(f"\ne1 · e2 = {dot_e1_e2}")
print(f"Are e1 and e2 perpendicular? {are_perpendicular(e1, e2)}")

# Non-perpendicular example
u = [1, 1, 0]
v = [1, 0, 0]
dot_uv = sum(u[i] * v[i] for i in range(len(u)))
print(f"\nOther vectors:")
print(f"u = {u}")  
print(f"v = {v}")
print(f"u · v = {dot_uv}")
print(f"Are u and v perpendicular? {are_perpendicular(u, v)}")


# Orthonormal vectors - perpendicular AND unit length
print("\nOrthonormal Vectors:")
print("-" * 19)

def vector_length(vec):
    return math.sqrt(sum(x*x for x in vec))

def make_unit_vector(vec):
    length = vector_length(vec)
    return [x/length for x in vec]

# Standard basis vectors are already orthonormal
print("Standard basis vectors:")
for i, vec in enumerate([e1, e2, e3]):
    length = vector_length(vec)
    print(f"e{i+1} = {vec}, length = {length}")

print("✓ Perpendicular to each other")  
print("✓ Each has length 1")
print("= ORTHONORMAL")

# Make custom orthonormal vectors
v1 = [1, 1, 0]
v1_unit = make_unit_vector(v1)
print(f"\nCustom vector: {v1}")
print(f"Made unit: {[round(x, 3) for x in v1_unit]}")
print(f"Length: {vector_length(v1_unit):.3f}")

# Gram-Schmidt process - make perpendicular vectors
print("\nGram-Schmidt Process:")
print("-" * 21)
print("Take non-perpendicular vectors, make them perpendicular")

# Start with two vectors that aren't perpendicular
a = [1, 1]
b = [1, 0]

print(f"Original vectors: a = {a}, b = {b}")
dot_ab = sum(a[i] * b[i] for i in range(len(a)))
print(f"a · b = {dot_ab} (not zero, so not perpendicular)")

# Step 1: Keep first vector as is
u1 = a[:]
print(f"\nStep 1: u1 = a = {u1}")

# Step 2: Remove component of b that's parallel to u1
# Formula: u2 = b - proj_u1(b)
# proj_u1(b) = (b·u1 / u1·u1) * u1

dot_b_u1 = sum(b[i] * u1[i] for i in range(len(b)))
dot_u1_u1 = sum(u1[i] * u1[i] for i in range(len(u1)))
projection_scalar = dot_b_u1 / dot_u1_u1

projection = [projection_scalar * u1[i] for i in range(len(u1))]
u2 = [b[i] - projection[i] for i in range(len(b))]

print(f"Step 2: Project b onto u1")
print(f"Projection of b onto u1: {projection}")
print(f"u2 = b - projection = {u2}")

# Verify they're perpendicular
dot_u1_u2 = sum(u1[i] * u2[i] for i in range(len(u1)))
print(f"\nCheck: u1 · u2 = {dot_u1_u2:.10f} (should be ~0)")
print("✓ Now they're perpendicular!")

# Make them unit vectors too (orthonormal)
e1_new = make_unit_vector(u1)
e2_new = make_unit_vector(u2)

print(f"\nMade unit vectors:")
print(f"e1 = {[round(x, 3) for x in e1_new]}")  
print(f"e2 = {[round(x, 3) for x in e2_new]}")
print("Now we have orthonormal vectors!")

# Rotation matrices - special orthogonal matrices
print("\nRotation Matrices:")
print("-" * 17)

def rotation_2d(angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return [[cos_a, -sin_a],
            [sin_a, cos_a]]

# 90 degree rotation
R90 = rotation_2d(90)
print(f"90° rotation matrix:")
for row in R90:
    print([round(x, 3) for x in row])

# Rotate a vector
vector = [1, 0]
rotated = [R90[0][0]*vector[0] + R90[0][1]*vector[1],
           R90[1][0]*vector[0] + R90[1][1]*vector[1]]

print(f"\nRotate {vector} by 90°: {[round(x, 3) for x in rotated]}")
print("Rotates [1,0] to [0,1] ✓")

print("\nDay 4 complete! ✅")
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
