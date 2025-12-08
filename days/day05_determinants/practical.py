# Day 5: Determinants
# A single number that tells us about a matrix
# Amey Prakash Sawant

# Determinants - like a signature of the matrix
print("Day 5: Determinants")
print("=" * 18)

# For 2x2 matrix, determinant = ad - bc
def det_2x2(matrix):
    # [[a, b],
    #  [c, d]]
    a = matrix[0][0]
    b = matrix[0][1] 
    c = matrix[1][0]
    d = matrix[1][1]
    return a * d - b * c

# Example 2x2 matrix
mat_2x2 = [[3, 2],
           [1, 4]]

det_value = det_2x2(mat_2x2)
print("2x2 Matrix:")
for row in mat_2x2:
    print(row)

print(f"\nDeterminant = {mat_2x2[0][0]}×{mat_2x2[1][1]} - {mat_2x2[0][1]}×{mat_2x2[1][0]}")
print(f"           = {mat_2x2[0][0]*mat_2x2[1][1]} - {mat_2x2[0][1]*mat_2x2[1][0]}")  
print(f"           = {det_value}")

# For 3x3 matrix, use cofactor expansion
def det_3x3(matrix):
    # Expand along first row
    a = matrix[0][0] 
    b = matrix[0][1]
    c = matrix[0][2]
    
    # 2x2 determinants (minors)
    minor1 = matrix[1][1]*matrix[2][2] - matrix[1][2]*matrix[2][1]
    minor2 = matrix[1][0]*matrix[2][2] - matrix[1][2]*matrix[2][0]  
    minor3 = matrix[1][0]*matrix[2][1] - matrix[1][1]*matrix[2][0]
    
    # Cofactor expansion: a*minor1 - b*minor2 + c*minor3
    return a*minor1 - b*minor2 + c*minor3

# Example 3x3 matrix
mat_3x3 = [[1, 2, 3],
           [4, 5, 6], 
           [7, 8, 10]]

det_3x3_value = det_3x3(mat_3x3)
print("\n3x3 Matrix:")
for row in mat_3x3:
    print(row)

print(f"\nDeterminant = {det_3x3_value}")

# What determinant tells us
print("\nWhat the determinant means:")
print("-" * 26)

# Zero determinant = no inverse (singular)
singular = [[1, 2],
            [2, 4]]  # Second row is 2× first row

det_singular = det_2x2(singular)
print(f"Matrix {singular}")
print(f"Determinant = {det_singular}")
print("Zero determinant = NO INVERSE = SINGULAR")

# Negative determinant = orientation change
negative_det = [[0, 1],
                [1, 0]]  # Swap x and y axes

det_negative = det_2x2(negative_det)
print(f"\nMatrix {negative_det}")
print(f"Determinant = {det_negative}")
print("Negative determinant = FLIPS ORIENTATION")

# Identity matrix
identity = [[1, 0],
            [0, 1]]

det_identity = det_2x2(identity)
print(f"\nIdentity matrix {identity}")
print(f"Determinant = {det_identity}")
print("Identity has determinant 1 (no change)")

print("\nDay 5 complete! ✅")

# Unit square transformation
def visualize_transformation(A, title):
    """Visualize how a matrix transforms the unit square"""
    # Unit square vertices
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    
    # Transform
    transformed = A @ square
    
    return square, transformed

# Different transformations
matrices = {
    'Scaling (2x)': np.array([[2, 0], [0, 2]]),
    'Shear': np.array([[1, 1], [0, 1]]),
    'Rotation (45°)': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                 [np.sin(np.pi/4), np.cos(np.pi/4)]]),
    'Reflection': np.array([[1, 0], [0, -1]]),
    'Singular': np.array([[1, 2], [0.5, 1]])
}

for name, A in matrices.items():
    det = np.linalg.det(A)
    print(f"{name}:")
    print(f"  Matrix:\n{A}")
    print(f"  det = {det:.4f}")
    print(f"  Area scaling: |det| = {abs(det):.4f}")
    print(f"  Orientation: {'preserved' if det > 0 else 'flipped' if det < 0 else 'collapsed'}")
    print()


# ============================================
# 3. PROPERTIES OF DETERMINANTS
# ============================================

print("=" * 50)
print("3. PROPERTIES OF DETERMINANTS")
print("=" * 50)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Property 1: det(I) = 1
I = np.eye(2)
print(f"det(I) = {np.linalg.det(I):.4f}")

# Property 2: det(A^T) = det(A)
print(f"\ndet(A) = {np.linalg.det(A):.4f}")
print(f"det(A^T) = {np.linalg.det(A.T):.4f}")

# Property 3: det(AB) = det(A) * det(B)
det_AB = np.linalg.det(A @ B)
det_A_times_det_B = np.linalg.det(A) * np.linalg.det(B)
print(f"\ndet(AB) = {det_AB:.4f}")
print(f"det(A) * det(B) = {det_A_times_det_B:.4f}")

# Property 4: det(A^-1) = 1/det(A)
A_inv = np.linalg.inv(A)
print(f"\ndet(A^-1) = {np.linalg.det(A_inv):.4f}")
print(f"1/det(A) = {1/np.linalg.det(A):.4f}")

# Property 5: det(cA) = c^n * det(A)
c = 2
n = 2
det_cA = np.linalg.det(c * A)
print(f"\ndet(2A) = {det_cA:.4f}")
print(f"2^2 * det(A) = {(c**n) * np.linalg.det(A):.4f}")


# ============================================
# 4. SINGULAR VS INVERTIBLE MATRICES
# ============================================

print("\n" + "=" * 50)
print("4. SINGULAR VS INVERTIBLE MATRICES")
print("=" * 50)

def check_invertibility(A):
    """Check if matrix is invertible using determinant"""
    det = np.linalg.det(A)
    rank = np.linalg.matrix_rank(A)
    
    print(f"Matrix:\n{A}")
    print(f"Determinant: {det:.6f}")
    print(f"Rank: {rank}")
    print(f"Invertible: {abs(det) > 1e-10}")
    return abs(det) > 1e-10

# Invertible matrix
print("--- Invertible Matrix ---")
A_inv = np.array([[4, 7], [2, 6]])
check_invertibility(A_inv)

# Singular matrix
print("\n--- Singular Matrix ---")
A_sing = np.array([[1, 2], [2, 4]])  # Row 2 = 2 * Row 1
check_invertibility(A_sing)


# ============================================
# 5. DETERMINANT AND EIGENVALUES
# ============================================

print("\n" + "=" * 50)
print("5. DETERMINANT AND EIGENVALUES")
print("=" * 50)

A = np.array([[4, 2],
              [1, 3]])

det_A = np.linalg.det(A)
eigenvalues = np.linalg.eigvals(A)
product_eigenvalues = np.prod(eigenvalues)

print(f"Matrix A:\n{A}")
print(f"\nDeterminant: {det_A:.4f}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Product of eigenvalues: {product_eigenvalues.real:.4f}")
print(f"det(A) = product of eigenvalues? {np.isclose(det_A, product_eigenvalues.real)}")


# ============================================
# 6. CRAMER'S RULE
# ============================================

print("\n" + "=" * 50)
print("6. CRAMER'S RULE")
print("=" * 50)

def cramers_rule(A, b):
    """Solve Ax = b using Cramer's Rule"""
    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-10:
        return None, "Matrix is singular"
    
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b  # Replace column i with b
        x[i] = np.linalg.det(Ai) / det_A
    
    return x, "Success"

# System: 2x + 3y = 8, 4x + y = 10
A = np.array([[2, 3],
              [4, 1]])
b = np.array([8, 10])

x_cramer, status = cramers_rule(A, b)
x_solve = np.linalg.solve(A, b)

print(f"System: Ax = b")
print(f"A:\n{A}")
print(f"b: {b}")
print(f"\nCramer's Rule solution: {x_cramer}")
print(f"np.linalg.solve: {x_solve}")
print(f"Solutions match: {np.allclose(x_cramer, x_solve)}")

# Verify
print(f"\nVerification A @ x: {A @ x_cramer}")


# ============================================
# 7. DETERMINANT IN GAUSSIAN DISTRIBUTION
# ============================================

print("\n" + "=" * 50)
print("7. DETERMINANT IN GAUSSIAN DISTRIBUTION")
print("=" * 50)

def multivariate_gaussian_pdf(x, mean, cov):
    """Compute multivariate Gaussian PDF"""
    n = len(x)
    det_cov = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    
    norm_factor = 1.0 / (np.sqrt((2 * np.pi)**n * det_cov))
    diff = x - mean
    exponent = -0.5 * diff @ cov_inv @ diff
    
    return norm_factor * np.exp(exponent), det_cov

# Example: 2D Gaussian
mean = np.array([0, 0])
cov = np.array([[1.0, 0.5],
                [0.5, 1.0]])

x = np.array([0.5, 0.5])
pdf, det_cov = multivariate_gaussian_pdf(x, mean, cov)

print(f"Covariance matrix:\n{cov}")
print(f"Determinant: {det_cov:.4f}")
print(f"Point x: {x}")
print(f"PDF value: {pdf:.6f}")
print(f"\nNote: Larger det → wider distribution → smaller PDF values")


# ============================================
# 8. ROW OPERATIONS EFFECT
# ============================================

print("\n" + "=" * 50)
print("8. ROW OPERATIONS EFFECT ON DETERMINANT")
print("=" * 50)

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])

det_original = np.linalg.det(A)
print(f"Original matrix:\n{A}")
print(f"det = {det_original:.4f}")

# Swap rows
A_swap = A.copy()
A_swap[[0, 1]] = A_swap[[1, 0]]
print(f"\nAfter swapping rows 1 and 2:\n{A_swap}")
print(f"det = {np.linalg.det(A_swap):.4f} (sign changed!)")

# Multiply row by scalar
A_scale = A.copy()
A_scale[0] = 2 * A_scale[0]
print(f"\nAfter multiplying row 1 by 2:\n{A_scale}")
print(f"det = {np.linalg.det(A_scale):.4f} (multiplied by 2)")

# Add row to another
A_add = A.copy()
A_add[1] = A_add[1] + A_add[0]
print(f"\nAfter adding row 1 to row 2:\n{A_add}")
print(f"det = {np.linalg.det(A_add):.4f} (unchanged!)")


# ============================================
# 9. VISUALIZATION
# ============================================

print("\n" + "=" * 50)
print("9. VISUALIZATION")
print("=" * 50)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Transformations to visualize
transformations = [
    ('Identity\ndet=1', np.eye(2)),
    ('Scale (2x)\ndet=4', np.array([[2, 0], [0, 2]])),
    ('Shear\ndet=1', np.array([[1, 0.5], [0, 1]])),
    ('Rotation (45°)\ndet=1', np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                         [np.sin(np.pi/4), np.cos(np.pi/4)]])),
    ('Reflection\ndet=-1', np.array([[1, 0], [0, -1]])),
    ('Singular\ndet=0', np.array([[1, 2], [0.5, 1]]))
]

# Unit square
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

for ax, (title, A) in zip(axes, transformations):
    det = np.linalg.det(A)
    
    # Original square
    ax.add_patch(plt.Polygon(square, fill=True, alpha=0.3, color='blue', label='Original'))
    
    # Transformed
    transformed = (A @ square.T).T
    ax.add_patch(plt.Polygon(transformed, fill=True, alpha=0.5, color='red', label='Transformed'))
    
    # Grid lines for original
    for i in range(2):
        ax.plot([square[i, 0], square[i+1, 0]], 
                [square[i, 1], square[i+1, 1]], 'b-', alpha=0.5)
    ax.plot([square[3, 0], square[0, 0]], 
            [square[3, 1], square[0, 1]], 'b-', alpha=0.5)
    
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(f'{title}\nArea: 1 → {abs(det):.2f}')
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('determinants_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'determinants_visualization.png'")


print("\n" + "=" * 50)
print("Day 5 Complete! 🎉")
print("=" * 50)
