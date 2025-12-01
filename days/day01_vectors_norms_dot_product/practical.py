"""
Day 1: Vectors, Norms, Dot Product & Projections
Practical Implementation in Python

Author: Amey Prakash Sawant
100 Days of AI/ML Journey
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. VECTORS - Creation and Basic Operations
# ============================================

print("=" * 50)
print("1. VECTORS")
print("=" * 50)

# Creating vectors
v1 = np.array([3, 4])
v2 = np.array([1, 2])
v3 = np.array([1, 2, 3, 4, 5])  # Higher dimensional

print(f"Vector v1: {v1}")
print(f"Vector v2: {v2}")
print(f"Vector v3 (5D): {v3}")

# Vector Addition
v_sum = v1 + v2
print(f"\nVector Addition (v1 + v2): {v_sum}")

# Scalar Multiplication
scalar = 3
v_scaled = scalar * v1
print(f"Scalar Multiplication (3 * v1): {v_scaled}")

# Element-wise operations
v_multiply = v1 * v2
print(f"Element-wise Multiplication: {v_multiply}")


# ============================================
# 2. NORMS - Measuring Vector Magnitude
# ============================================

print("\n" + "=" * 50)
print("2. NORMS")
print("=" * 50)

v = np.array([3, -4, 5])

# L1 Norm (Manhattan)
l1_norm = np.linalg.norm(v, ord=1)
l1_manual = np.sum(np.abs(v))
print(f"L1 Norm of {v}: {l1_norm}")
print(f"L1 Norm (manual): {l1_manual}")

# L2 Norm (Euclidean)
l2_norm = np.linalg.norm(v, ord=2)
l2_manual = np.sqrt(np.sum(v**2))
print(f"\nL2 Norm of {v}: {l2_norm}")
print(f"L2 Norm (manual): {l2_manual}")

# L-infinity Norm (Max)
linf_norm = np.linalg.norm(v, ord=np.inf)
linf_manual = np.max(np.abs(v))
print(f"\nL∞ Norm of {v}: {linf_norm}")
print(f"L∞ Norm (manual): {linf_manual}")

# Unit Vector (normalizing a vector)
unit_v = v / np.linalg.norm(v)
print(f"\nUnit vector of {v}: {unit_v}")
print(f"Magnitude of unit vector: {np.linalg.norm(unit_v):.4f}")


# ============================================
# 3. DOT PRODUCT
# ============================================

print("\n" + "=" * 50)
print("3. DOT PRODUCT")
print("=" * 50)

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Different ways to compute dot product
dot1 = np.dot(u, v)
dot2 = u @ v
dot3 = np.sum(u * v)

print(f"u = {u}")
print(f"v = {v}")
print(f"Dot product (np.dot): {dot1}")
print(f"Dot product (@ operator): {dot2}")
print(f"Dot product (manual): {dot3}")

# Geometric interpretation
def angle_between_vectors(u, v):
    """Calculate angle between two vectors in degrees"""
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    # Clip to handle numerical errors
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    return theta_deg

angle = angle_between_vectors(u, v)
print(f"\nAngle between u and v: {angle:.2f} degrees")

# Check orthogonality
orthogonal_v1 = np.array([1, 0])
orthogonal_v2 = np.array([0, 1])
print(f"\nOrthogonal check: {orthogonal_v1} · {orthogonal_v2} = {np.dot(orthogonal_v1, orthogonal_v2)}")


# ============================================
# 4. VECTOR PROJECTIONS
# ============================================

print("\n" + "=" * 50)
print("4. VECTOR PROJECTIONS")
print("=" * 50)

u = np.array([3, 4])
v = np.array([5, 0])

# Scalar projection (length of shadow)
scalar_proj = np.dot(u, v) / np.linalg.norm(v)
print(f"u = {u}")
print(f"v = {v}")
print(f"Scalar projection of u onto v: {scalar_proj}")

# Vector projection (the actual shadow vector)
vector_proj = (np.dot(u, v) / np.dot(v, v)) * v
print(f"Vector projection of u onto v: {vector_proj}")

# Verify: the difference should be orthogonal to v
residual = u - vector_proj
print(f"Residual (u - proj): {residual}")
print(f"Residual · v (should be ~0): {np.dot(residual, v):.10f}")


def project_vector(u, v):
    """Project vector u onto vector v"""
    return (np.dot(u, v) / np.dot(v, v)) * v


# ============================================
# 5. COSINE SIMILARITY
# ============================================

print("\n" + "=" * 50)
print("5. COSINE SIMILARITY")
print("=" * 50)

def cosine_similarity(u, v):
    """Calculate cosine similarity between two vectors"""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Example: Document similarity
# Imagine these are word count vectors for documents
doc1 = np.array([3, 2, 0, 5, 0, 0, 1])  # Document 1
doc2 = np.array([3, 1, 0, 4, 0, 0, 2])  # Document 2 (similar)
doc3 = np.array([0, 0, 5, 0, 3, 4, 0])  # Document 3 (different topic)

sim_12 = cosine_similarity(doc1, doc2)
sim_13 = cosine_similarity(doc1, doc3)
sim_23 = cosine_similarity(doc2, doc3)

print("Document Similarity using Cosine Similarity:")
print(f"Similarity(doc1, doc2): {sim_12:.4f}")
print(f"Similarity(doc1, doc3): {sim_13:.4f}")
print(f"Similarity(doc2, doc3): {sim_23:.4f}")


# ============================================
# 6. VISUALIZATION
# ============================================

print("\n" + "=" * 50)
print("6. VISUALIZATION")
print("=" * 50)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Vector Addition
ax1 = axes[0]
u = np.array([2, 1])
v = np.array([1, 2])
ax1.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='r', label='u')
ax1.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='v')
ax1.quiver(0, 0, u[0]+v[0], u[1]+v[1], angles='xy', scale_units='xy', scale=1, color='g', label='u+v')
ax1.quiver(u[0], u[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', alpha=0.5)
ax1.set_xlim(-0.5, 4)
ax1.set_ylim(-0.5, 4)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.legend()
ax1.set_title('Vector Addition')

# Plot 2: Vector Projection
ax2 = axes[1]
u = np.array([3, 4])
v = np.array([5, 1])
proj = project_vector(u, v)
ax2.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='r', label='u')
ax2.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='v')
ax2.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, color='g', label='proj_v(u)')
ax2.plot([u[0], proj[0]], [u[1], proj[1]], 'k--', alpha=0.5)
ax2.set_xlim(-0.5, 6)
ax2.set_ylim(-0.5, 5)
ax2.set_aspect('equal')
ax2.grid(True)
ax2.legend()
ax2.set_title('Vector Projection')

# Plot 3: Unit Circle and Norms
ax3 = axes[2]
theta = np.linspace(0, 2*np.pi, 100)
# L2 norm unit circle
ax3.plot(np.cos(theta), np.sin(theta), 'b-', label='L2 unit circle')
# L1 norm unit "circle" (diamond)
l1_x = [1, 0, -1, 0, 1]
l1_y = [0, 1, 0, -1, 0]
ax3.plot(l1_x, l1_y, 'r-', label='L1 unit "circle"')
# L-inf norm unit "circle" (square)
linf_x = [1, 1, -1, -1, 1]
linf_y = [1, -1, -1, 1, 1]
ax3.plot(linf_x, linf_y, 'g-', label='L∞ unit "circle"')
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_aspect('equal')
ax3.grid(True)
ax3.legend()
ax3.set_title('Different Norm "Unit Circles"')

plt.tight_layout()
plt.savefig('vectors_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'vectors_visualization.png'")


# ============================================
# 7. ML APPLICATION: Simple Similarity Search
# ============================================

print("\n" + "=" * 50)
print("7. ML APPLICATION: Similarity Search")
print("=" * 50)

# Simulated word embeddings (like Word2Vec)
embeddings = {
    'king': np.array([0.8, 0.2, 0.9, 0.1]),
    'queen': np.array([0.7, 0.8, 0.9, 0.1]),
    'man': np.array([0.9, 0.1, 0.1, 0.2]),
    'woman': np.array([0.8, 0.9, 0.1, 0.2]),
    'apple': np.array([0.1, 0.1, 0.2, 0.9]),
    'orange': np.array([0.15, 0.1, 0.25, 0.85]),
}

def find_most_similar(word, embeddings, top_n=3):
    """Find most similar words using cosine similarity"""
    target = embeddings[word]
    similarities = []
    
    for other_word, other_vec in embeddings.items():
        if other_word != word:
            sim = cosine_similarity(target, other_vec)
            similarities.append((other_word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

print("Finding similar words using cosine similarity:")
for word in ['king', 'apple']:
    similar = find_most_similar(word, embeddings)
    print(f"\nMost similar to '{word}':")
    for similar_word, sim in similar:
        print(f"  {similar_word}: {sim:.4f}")

# Famous word analogy: king - man + woman ≈ queen
result = embeddings['king'] - embeddings['man'] + embeddings['woman']
print("\n\nWord Analogy: king - man + woman = ?")
best_match = None
best_sim = -1
for word, vec in embeddings.items():
    if word not in ['king', 'man', 'woman']:
        sim = cosine_similarity(result, vec)
        if sim > best_sim:
            best_sim = sim
            best_match = word
print(f"Answer: {best_match} (similarity: {best_sim:.4f})")


print("\n" + "=" * 50)
print("Day 1 Complete! 🎉")
print("=" * 50)
