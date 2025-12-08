"""
Project 2: PCA From Scratch
Implementing Principal Component Analysis using pure linear algebra.
"""

import numpy as np
import matplotlib.pyplot as plt


class PCAFromScratch:
    """Principal Component Analysis implementation from scratch."""
    
    def __init__(self, n_components=None):
        """
        Initialize PCA.
        
        Parameters:
        -----------
        n_components : int or None
            Number of principal components to keep.
            If None, keep all components.
        """
        self.n_components = n_components
        self.components_ = None  # Principal components (eigenvectors)
        self.explained_variance_ = None  # Eigenvalues
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_ = None
        
    def fit(self, X):
        """
        Fit PCA to data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute covariance matrix
        # Cov = (1/(n-1)) * X^T @ X
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Step 4: Sort by eigenvalue (descending order)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Step 5: Select top n_components
        if self.n_components is None:
            self.n_components = n_features
        
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / np.sum(eigenvalues)
        )
        
        return self
    
    def transform(self, X):
        """
        Transform data to principal component space.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array of shape (n_samples, n_components)
        """
        X = np.array(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space.
        
        Parameters:
        -----------
        X_transformed : array of shape (n_samples, n_components)
            Data in principal component space
            
        Returns:
        --------
        X_original : array of shape (n_samples, n_features)
        """
        return X_transformed @ self.components_ + self.mean_
    
    def get_reconstruction_error(self, X):
        """Calculate reconstruction error (MSE)."""
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return np.mean((X - X_reconstructed) ** 2)


class PCAUsingSVD:
    """PCA implementation using Singular Value Decomposition."""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.singular_values_ = None
        
    def fit(self, X):
        """Fit PCA using SVD."""
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Singular values to eigenvalues
        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = np.sum(explained_variance)
        
        # Select components
        if self.n_components is None:
            self.n_components = n_features
            
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]
        self.explained_variance_ = explained_variance[:self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
        )
        
        return self
    
    def transform(self, X):
        """Project data onto principal components."""
        X = np.array(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


def generate_sample_data(n_samples=500):
    """Generate sample 3D data with clear structure."""
    np.random.seed(42)
    
    # Create data with one dominant direction
    t = np.linspace(0, 4*np.pi, n_samples)
    
    # Main direction
    x = t + np.random.randn(n_samples) * 0.3
    y = 0.5 * t + np.random.randn(n_samples) * 0.5
    z = 0.2 * t + np.random.randn(n_samples) * 0.8
    
    return np.column_stack([x, y, z])


def visualize_pca_2d():
    """Visualize PCA on 2D data."""
    # Generate 2D data
    np.random.seed(42)
    n = 200
    
    # Create correlated data
    mean = [2, 3]
    cov = [[2, 1.5], [1.5, 1.5]]
    X = np.random.multivariate_normal(mean, cov, n)
    
    # Fit PCA
    pca = PCAFromScratch(n_components=2)
    pca.fit(X)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original data with principal components
    ax1 = axes[0]
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Data')
    
    # Plot principal component directions
    origin = pca.mean_
    for i, (component, var) in enumerate(zip(
        pca.components_, pca.explained_variance_
    )):
        # Scale by explained variance for visualization
        scale = np.sqrt(var) * 2
        ax1.annotate(
            '', xy=origin + component * scale,
            xytext=origin,
            arrowprops=dict(arrowstyle='->', color=f'C{i+1}', lw=3)
        )
        ax1.annotate(
            f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})',
            xy=origin + component * scale,
            fontsize=12, fontweight='bold', color=f'C{i+1}'
        )
    
    ax1.scatter(*origin, c='red', s=100, marker='x', 
                label='Center', linewidths=3)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_title('Original Data with Principal Components')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Transformed data
    X_transformed = pca.transform(X)
    ax2 = axes[1]
    ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Data in Principal Component Space')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_2d_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_pca_3d_to_2d():
    """Visualize dimensionality reduction from 3D to 2D."""
    # Generate 3D data
    X = generate_sample_data(300)
    
    # Fit PCA to reduce to 2D
    pca = PCAFromScratch(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # Reconstruction
    X_reconstructed = pca.inverse_transform(X_2d)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Original 3D data
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.arange(len(X)), 
                cmap='viridis', alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original 3D Data')
    
    # 2D projection
    ax2 = fig.add_subplot(132)
    scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=np.arange(len(X)), 
                          cmap='viridis', alpha=0.6)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title(f'2D Projection\n({sum(pca.explained_variance_ratio_):.1%} variance)')
    ax2.grid(True, alpha=0.3)
    
    # Reconstructed 3D
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], 
                X_reconstructed[:, 2], c=np.arange(len(X)), 
                cmap='viridis', alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'Reconstructed 3D\n(MSE: {pca.get_reconstruction_error(X):.4f})')
    
    plt.tight_layout()
    plt.savefig('pca_3d_to_2d.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_pca_methods():
    """Compare eigenvalue-based PCA with SVD-based PCA."""
    X = generate_sample_data(200)
    
    # Fit both methods
    pca_eig = PCAFromScratch(n_components=2)
    pca_svd = PCAUsingSVD(n_components=2)
    
    X_eig = pca_eig.fit_transform(X)
    X_svd = pca_svd.fit_transform(X)
    
    print("=" * 60)
    print("Comparison: Eigenvalue vs SVD PCA")
    print("=" * 60)
    
    print("\n📊 Explained Variance Ratio:")
    print(f"  Eigenvalue method: {pca_eig.explained_variance_ratio_}")
    print(f"  SVD method:        {pca_svd.explained_variance_ratio_}")
    
    print("\n📐 Principal Components (normalized):")
    for i in range(2):
        print(f"\n  PC{i+1}:")
        print(f"    Eigenvalue: {pca_eig.components_[i]}")
        print(f"    SVD:        {pca_svd.components_[i]}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(X_eig[:, 0], X_eig[:, 1], alpha=0.5)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('Eigenvalue-based PCA')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(X_svd[:, 0], X_svd[:, 1], alpha=0.5)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('SVD-based PCA')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_method_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_cumulative_variance():
    """Visualize cumulative explained variance (scree plot)."""
    # Generate higher dimensional data
    np.random.seed(42)
    n_samples, n_features = 500, 10
    
    # Create data with decreasing variance in each dimension
    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        variance = 10 / (i + 1)  # Decreasing variance
        X[:, i] = np.random.randn(n_samples) * np.sqrt(variance)
    
    # Add some correlations
    X[:, 1] += 0.5 * X[:, 0]
    X[:, 2] += 0.3 * X[:, 0] + 0.2 * X[:, 1]
    
    # Fit full PCA
    pca = PCAFromScratch(n_components=n_features)
    pca.fit(X)
    
    # Calculate cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    ax1 = axes[0]
    x = np.arange(1, n_features + 1)
    bars = ax1.bar(x, pca.explained_variance_ratio_, alpha=0.7, 
                   color='steelblue', edgecolor='black')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot')
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, ratio in zip(bars, pca.explained_variance_ratio_):
        if ratio > 0.03:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ratio:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Cumulative variance
    ax2 = axes[1]
    ax2.plot(x, cumulative_variance, 'o-', color='steelblue', 
             linewidth=2, markersize=8)
    ax2.axhline(y=0.95, color='red', linestyle='--', 
                label='95% threshold', alpha=0.7)
    ax2.axhline(y=0.90, color='orange', linestyle='--', 
                label='90% threshold', alpha=0.7)
    ax2.fill_between(x, 0, cumulative_variance, alpha=0.3)
    
    # Find components needed for 95%
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    ax2.axvline(x=n_95, color='red', linestyle=':', alpha=0.7)
    ax2.annotate(f'{n_95} components\nfor 95%', xy=(n_95, 0.95),
                xytext=(n_95 + 1, 0.85), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))
    
    ax2.set_xlabel('Number of Principal Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Explained Variance')
    ax2.set_xticks(x)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_variance_explained.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n📈 Variance Analysis:")
    print(f"  Components for 90% variance: {np.argmax(cumulative_variance >= 0.90) + 1}")
    print(f"  Components for 95% variance: {n_95}")
    print(f"  Components for 99% variance: {np.argmax(cumulative_variance >= 0.99) + 1}")


def demo_image_compression():
    """Demonstrate PCA for image compression (using synthetic image)."""
    # Create a synthetic grayscale image
    np.random.seed(42)
    size = 64
    
    # Create image with some structure
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    image = (
        np.sin(5 * np.pi * x) * np.cos(5 * np.pi * y) +
        0.5 * np.sin(10 * np.pi * x) +
        0.3 * np.random.randn(size, size)
    )
    image = (image - image.min()) / (image.max() - image.min())
    
    # Apply PCA for different number of components
    n_components_list = [5, 10, 20, 32]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title(f'Original\n({size}x{size} = {size*size} values)')
    axes[0, 0].axis('off')
    
    # Compressed versions
    for idx, n_comp in enumerate(n_components_list):
        pca = PCAFromScratch(n_components=n_comp)
        
        # Treat each row as a sample, columns as features
        compressed = pca.fit_transform(image)
        reconstructed = pca.inverse_transform(compressed)
        
        compression_ratio = (size * size) / (n_comp * size + n_comp * size)
        mse = np.mean((image - reconstructed) ** 2)
        
        row = 0 if idx < 2 else 1
        col = (idx % 2) + 1 if idx < 2 else idx - 2
        
        axes[row, col].imshow(reconstructed, cmap='gray')
        axes[row, col].set_title(
            f'{n_comp} components\n'
            f'Compression: {compression_ratio:.1f}x, MSE: {mse:.4f}'
        )
        axes[row, col].axis('off')
    
    # Empty subplot for explanation
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 
                    "PCA Image Compression\n\n"
                    "• Each row treated as a sample\n"
                    "• Columns are features\n"
                    "• Keep top-k components\n"
                    "• Reconstruct from reduced data",
                    ha='center', va='center', fontsize=12,
                    transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('pca_image_compression.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Project 2: PCA From Scratch")
    print("=" * 60)
    
    # Basic demonstration
    print("\n" + "=" * 60)
    print("1. Basic PCA Demonstration")
    print("=" * 60)
    
    X = generate_sample_data(100)
    pca = PCAFromScratch(n_components=2)
    pca.fit(X)
    
    print(f"\n📊 Data shape: {X.shape}")
    print(f"📊 Principal components shape: {pca.components_.shape}")
    print(f"\n📈 Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"📈 Total explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Visualizations
    print("\n" + "=" * 60)
    print("2. 2D Visualization")
    print("=" * 60)
    visualize_pca_2d()
    
    print("\n" + "=" * 60)
    print("3. 3D to 2D Dimensionality Reduction")
    print("=" * 60)
    visualize_pca_3d_to_2d()
    
    print("\n" + "=" * 60)
    print("4. Method Comparison (Eigenvalue vs SVD)")
    print("=" * 60)
    compare_pca_methods()
    
    print("\n" + "=" * 60)
    print("5. Cumulative Variance (Scree Plot)")
    print("=" * 60)
    visualize_cumulative_variance()
    
    print("\n" + "=" * 60)
    print("6. Image Compression Demo")
    print("=" * 60)
    demo_image_compression()
    
    print("\n" + "=" * 60)
    print("✅ Project 2 Complete!")
    print("=" * 60)
