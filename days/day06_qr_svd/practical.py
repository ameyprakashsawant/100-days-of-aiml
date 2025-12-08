# Day 6: QR and SVD
# Breaking down matrices in useful ways
# Amey Prakash Sawant

print("Day 6: QR Decomposition & SVD")
print("=" * 29)

# QR Decomposition - breaking matrix into Q (orthogonal) and R (triangular)
# Think of it as reorganizing data

# Simple example matrix
A = [[1, 2],
     [3, 4],
     [5, 6]]

print("Matrix A (3x2):")
for row in A:
    print(row)

# For QR decomposition, we need orthogonal vectors
# Let's do a simple Gram-Schmidt like Day 4

def normalize_vector(v):
    # Make vector length 1
    length = 0
    for x in v:
        length += x * x
    length = length ** 0.5
    
    return [x / length for x in v]

def dot_product(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result

def subtract_vectors(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]

def multiply_vector(v, scalar):
    return [x * scalar for x in v]

# Get columns of A
col1 = [A[i][0] for i in range(len(A))]  # [1, 3, 5]
col2 = [A[i][1] for i in range(len(A))]  # [2, 4, 6]

print(f"\nColumn 1: {col1}")
print(f"Column 2: {col2}")

# Gram-Schmidt to get orthogonal vectors
q1 = normalize_vector(col1)
print(f"\nOrthonormal vector 1: {[round(x, 3) for x in q1]}")

# Make col2 perpendicular to q1
projection = multiply_vector(q1, dot_product(col2, q1))
col2_perp = subtract_vectors(col2, projection)
q2 = normalize_vector(col2_perp)
print(f"Orthonormal vector 2: {[round(x, 3) for x in q2]}")

print("\nThis is the idea behind QR decomposition!")
print("Q = orthogonal matrix, R = triangular matrix")

print("\n" + "-" * 30)

# SVD - Singular Value Decomposition
# Think of it as finding the "principal directions" of data
print("SVD - Finding main directions in data")

# Example: ratings matrix (users × movies)
ratings = [[5, 3, 0, 1],
           [4, 0, 0, 1], 
           [1, 1, 0, 5],
           [1, 0, 0, 4],
           [0, 1, 5, 4]]

print("\nUser-Movie ratings matrix:")
print("Users → rows, Movies → columns")
for i, row in enumerate(ratings):
    print(f"User {i+1}: {row}")

print("\nSVD finds:")
print("1. Main patterns in user preferences")
print("2. Main patterns in movie types") 
print("3. How strong each pattern is")

print("\nExample patterns SVD might find:")
print("- Pattern 1: Action movie lovers vs Drama lovers")
print("- Pattern 2: Popular movies vs Niche movies")
print("- Pattern 3: New movies vs Classic movies")

# Simple example of how SVD helps
print("\nHow SVD helps:")
print("- Compress data (keep main patterns)")
print("- Remove noise")
print("- Find similar users/movies")
print("- Make recommendations")

# Image compression example concept
print("\nSVD in image compression:")
print("- Image = matrix of pixel values")
print("- SVD finds main 'features' of image")
print("- Keep top features = compressed image")
print("- Throw away small features = less storage")

print("\nDay 6 complete! ✅")
print("\nKey takeaways:")
print("- QR: Good for solving equations accurately")
print("- SVD: Good for finding patterns and compression")
print("- Both help us understand matrix structure")
print("=" * 50)

# Generate data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 3)
true_weights = np.array([2, -1, 0.5])
y = X @ true_weights + np.random.randn(n_samples) * 0.1

# Method 1: Normal equations (less stable)
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
theta_normal = XtX_inv @ X.T @ y

# Method 2: QR decomposition (more stable)
Q, R = np.linalg.qr(X)
theta_qr = np.linalg.solve(R, Q.T @ y)

# Method 3: NumPy lstsq (uses SVD internally)
theta_lstsq, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

print(f"True weights: {true_weights}")
print(f"Normal equations: {theta_normal.round(4)}")
print(f"QR decomposition: {theta_qr.round(4)}")
print(f"np.linalg.lstsq: {theta_lstsq.round(4)}")


# ============================================
# 3. SVD DECOMPOSITION
# ============================================

print("\n" + "=" * 50)
print("3. SVD DECOMPOSITION")
print("=" * 50)

A = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=float)

print(f"Matrix A (3x2):\n{A}")

# Full SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)
print(f"\nFull SVD:")
print(f"U (3x3):\n{U.round(4)}")
print(f"Singular values: {S.round(4)}")
print(f"V^T (2x2):\n{Vt.round(4)}")

# Reconstruct
# Need to create Sigma matrix
Sigma = np.zeros((3, 2))
Sigma[:2, :2] = np.diag(S)
reconstructed = U @ Sigma @ Vt
print(f"\nReconstructed U @ Σ @ V^T:\n{reconstructed.round(4)}")
print(f"Matches original? {np.allclose(reconstructed, A)}")

# Reduced SVD (more memory efficient)
U_reduced, S_reduced, Vt_reduced = np.linalg.svd(A, full_matrices=False)
print(f"\nReduced SVD:")
print(f"U (3x2):\n{U_reduced.round(4)}")
print(f"S: {S_reduced.round(4)}")
print(f"V^T (2x2):\n{Vt_reduced.round(4)}")


# ============================================
# 4. SVD PROPERTIES
# ============================================

print("\n" + "=" * 50)
print("4. SVD PROPERTIES")
print("=" * 50)

A = np.array([[1, 2, 0],
              [0, 1, 2],
              [1, 1, 1]], dtype=float)

U, S, Vt = np.linalg.svd(A)

print(f"Matrix A:\n{A}")
print(f"Singular values: {S.round(4)}")

# Rank
rank = np.sum(S > 1e-10)
print(f"\nRank (from SVD): {rank}")
print(f"Rank (from matrix_rank): {np.linalg.matrix_rank(A)}")

# Matrix norms
print(f"\n2-Norm (largest singular value): {S[0]:.4f}")
print(f"2-Norm (numpy): {np.linalg.norm(A, 2):.4f}")

frobenius_svd = np.sqrt(np.sum(S**2))
print(f"\nFrobenius norm (from SVD): {frobenius_svd:.4f}")
print(f"Frobenius norm (numpy): {np.linalg.norm(A, 'fro'):.4f}")

# Condition number
cond = S[0] / S[-1]
print(f"\nCondition number (from SVD): {cond:.4f}")
print(f"Condition number (numpy): {np.linalg.cond(A):.4f}")


# ============================================
# 5. LOW-RANK APPROXIMATION
# ============================================

print("\n" + "=" * 50)
print("5. LOW-RANK APPROXIMATION")
print("=" * 50)

# Create a matrix with low intrinsic rank
np.random.seed(42)
m, n, true_rank = 50, 40, 3

# Low rank matrix = UV^T where U is m×r and V is n×r
U_true = np.random.randn(m, true_rank)
V_true = np.random.randn(n, true_rank)
A = U_true @ V_true.T + np.random.randn(m, n) * 0.1  # Add small noise

print(f"Original matrix shape: {A.shape}")
print(f"True rank: {true_rank}")

# SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"Singular values (top 10): {S[:10].round(4)}")

# Approximation with different ranks
def low_rank_approx(U, S, Vt, k):
    """Compute rank-k approximation"""
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

for k in [1, 2, 3, 5, 10]:
    A_approx = low_rank_approx(U, S, Vt, k)
    error = np.linalg.norm(A - A_approx, 'fro')
    print(f"Rank-{k} approximation error: {error:.4f}")


# ============================================
# 6. IMAGE COMPRESSION WITH SVD
# ============================================

print("\n" + "=" * 50)
print("6. IMAGE COMPRESSION WITH SVD")
print("=" * 50)

# Create a sample grayscale image (gradient pattern)
img = np.zeros((256, 256))
for i in range(256):
    for j in range(256):
        img[i, j] = (i + j) / 2 + 50 * np.sin(i/10) * np.cos(j/10)

img = img / img.max() * 255  # Normalize to 0-255

print(f"Image shape: {img.shape}")
print(f"Original size (bytes): {img.nbytes}")

# SVD compression
U, S, Vt = np.linalg.svd(img, full_matrices=False)
print(f"Singular values range: {S[0]:.2f} to {S[-1]:.6f}")

def compress_image_svd(U, S, Vt, k):
    """Compress image keeping top k singular values"""
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Different compression levels
ks = [5, 10, 20, 50]
print("\nCompression results:")
for k in ks:
    compressed = compress_image_svd(U, S, Vt, k)
    error = np.linalg.norm(img - compressed, 'fro') / np.linalg.norm(img, 'fro')
    # Storage: k * (m + n + 1) vs m * n
    storage_ratio = k * (256 + 256 + 1) / (256 * 256)
    print(f"k={k:3d}: relative error={error:.4f}, storage={storage_ratio*100:.1f}%")


# ============================================
# 7. PCA WITH SVD
# ============================================

print("\n" + "=" * 50)
print("7. PCA WITH SVD")
print("=" * 50)

# Generate data with structure
np.random.seed(42)
n_samples = 200
# Data lies mostly in a 2D plane in 3D space
t = np.random.randn(n_samples, 2)
X = np.column_stack([
    t[:, 0] + 0.5 * t[:, 1],
    t[:, 1],
    0.1 * np.random.randn(n_samples)  # Small variance in 3rd dimension
])

print(f"Data shape: {X.shape}")

# Center the data
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

print(f"\nSingular values: {S.round(4)}")
print(f"Explained variance ratios: {(S**2 / np.sum(S**2)).round(4)}")

# Principal components are rows of V^T (or columns of V)
print(f"\nPrincipal components (V^T):\n{Vt.round(4)}")

# Project to 2D
n_components = 2
X_pca = X_centered @ Vt[:n_components].T
print(f"\nProjected data shape: {X_pca.shape}")


# ============================================
# 8. PSEUDOINVERSE WITH SVD
# ============================================

print("\n" + "=" * 50)
print("8. PSEUDOINVERSE WITH SVD")
print("=" * 50)

A = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=float)

# Using SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Pseudoinverse: V @ S^+ @ U^T
S_pinv = 1 / S
A_pinv_svd = Vt.T @ np.diag(S_pinv) @ U.T

# Using numpy
A_pinv_numpy = np.linalg.pinv(A)

print(f"Matrix A:\n{A}")
print(f"\nPseudoinverse (SVD):\n{A_pinv_svd.round(4)}")
print(f"\nPseudoinverse (numpy):\n{A_pinv_numpy.round(4)}")
print(f"\nMatch? {np.allclose(A_pinv_svd, A_pinv_numpy)}")

# Verify: A @ A^+ @ A = A
print(f"\nA @ A^+ @ A:\n{(A @ A_pinv_numpy @ A).round(4)}")


# ============================================
# 9. VISUALIZATION
# ============================================

print("\n" + "=" * 50)
print("9. VISUALIZATION")
print("=" * 50)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Singular value decay
ax1 = axes[0, 0]
# Create matrix with different rank structures
matrices = {
    'Rank 3': np.random.randn(50, 3) @ np.random.randn(3, 50),
    'Rank 10': np.random.randn(50, 10) @ np.random.randn(10, 50),
    'Full rank + noise': np.random.randn(50, 50)
}
for name, M in matrices.items():
    U, S, Vt = np.linalg.svd(M)
    ax1.semilogy(S[:30], label=name)
ax1.set_xlabel('Index')
ax1.set_ylabel('Singular Value (log)')
ax1.set_title('Singular Value Decay')
ax1.legend()
ax1.grid(True)

# Plot 2: Low rank approximation error
ax2 = axes[0, 1]
A_test = np.random.randn(100, 50) @ np.random.randn(50, 5) @ np.random.randn(5, 80)
U, S, Vt = np.linalg.svd(A_test, full_matrices=False)
errors = []
ks = range(1, 20)
for k in ks:
    A_approx = low_rank_approx(U, S, Vt, k)
    errors.append(np.linalg.norm(A_test - A_approx, 'fro'))
ax2.plot(ks, errors, 'b-o')
ax2.set_xlabel('Rank k')
ax2.set_ylabel('Frobenius Error')
ax2.set_title('Low-Rank Approximation Error')
ax2.grid(True)

# Plot 3: Image compression
ax3 = axes[0, 2]
ks_img = [5, 20, 50]
for i, k in enumerate(ks_img):
    compressed = compress_image_svd(U[:, :k], S[:k], Vt[:k, :], k)
    if i == 0:
        ax3.imshow(compressed, cmap='gray', alpha=0.33, label=f'k={k}')
    elif i == 1:
        ax3.imshow(compressed, cmap='gray', alpha=0.33)
    else:
        ax3.imshow(compressed, cmap='gray', alpha=0.33)
ax3.set_title('Image Compression: k=5, 20, 50')
ax3.axis('off')

# Plot 4: PCA visualization
ax4 = axes[1, 0]
ax4.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
ax4.set_title('PCA Projection (SVD)')
ax4.grid(True)

# Plot 5: QR vs Normal equations stability
ax5 = axes[1, 1]
condition_numbers = np.logspace(0, 6, 20)
errors_normal = []
errors_qr = []

for cond in condition_numbers:
    # Create ill-conditioned problem
    X = np.random.randn(100, 10)
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    S = np.logspace(0, -np.log10(cond), 10)
    X = U @ np.diag(S) @ Vt
    
    true_theta = np.random.randn(10)
    y = X @ true_theta
    
    # Normal equations
    try:
        theta_normal = np.linalg.inv(X.T @ X) @ X.T @ y
        errors_normal.append(np.linalg.norm(theta_normal - true_theta))
    except:
        errors_normal.append(np.nan)
    
    # QR
    Q, R = np.linalg.qr(X)
    theta_qr = np.linalg.solve(R, Q.T @ y)
    errors_qr.append(np.linalg.norm(theta_qr - true_theta))

ax5.loglog(condition_numbers, errors_normal, 'r-', label='Normal Equations')
ax5.loglog(condition_numbers, errors_qr, 'b-', label='QR')
ax5.set_xlabel('Condition Number')
ax5.set_ylabel('Error')
ax5.set_title('Numerical Stability: QR vs Normal')
ax5.legend()
ax5.grid(True)

# Plot 6: SVD geometric interpretation
ax6 = axes[1, 2]
# Unit circle
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# Transformation
A = np.array([[2, 1], [1, 1]])
ellipse = A @ circle

# SVD
U, S, Vt = np.linalg.svd(A)

ax6.plot(circle[0], circle[1], 'b--', label='Unit circle')
ax6.plot(ellipse[0], ellipse[1], 'r-', label='Transformed')

# Plot singular vectors
origin = [0, 0]
ax6.quiver(*origin, S[0]*U[0, 0], S[0]*U[1, 0], color='green', scale=1, 
           scale_units='xy', label=f'σ₁={S[0]:.2f}')
ax6.quiver(*origin, S[1]*U[0, 1], S[1]*U[1, 1], color='purple', scale=1,
           scale_units='xy', label=f'σ₂={S[1]:.2f}')

ax6.set_xlim(-3, 3)
ax6.set_ylim(-2, 3)
ax6.set_aspect('equal')
ax6.legend()
ax6.set_title('SVD: Circle to Ellipse')
ax6.grid(True)

plt.tight_layout()
plt.savefig('qr_svd_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'qr_svd_visualization.png'")


print("\n" + "=" * 50)
print("Day 6 Complete! 🎉")
print("=" * 50)
