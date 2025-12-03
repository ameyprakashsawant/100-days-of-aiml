# Image Transformer - Day 1 Project
# Simple image transformations using matrices
# Amey Prakash Sawant - 100 Days ML Journey

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def load_image(path):
    """Load an image and convert to numpy array"""
    img = Image.open(path)
    return np.array(img)

def create_test_image():
    """Make a simple test image with shapes"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Red square
    img[20:80, 20:80, 0] = 255
    
    # Green triangle (rough)
    for i in range(30):
        start_col = 50 + i//2
        end_col = start_col + 1
        img[10+i, start_col:end_col, 1] = 255
    
    # Blue circle
    center_x, center_y = 70, 70
    radius = 15
    
    for x in range(100):
        for y in range(100):
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            if distance <= radius:
                img[y, x, 2] = 255
    
    return img

def rotate_image(img, angle_deg):
    """Rotate image using rotation matrix"""
    height, width = img.shape[:2]
    
    # Convert angle to radians
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Create rotation matrix
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Create output image
    if len(img.shape) == 3:
        output = np.zeros_like(img)
    else:
        output = np.zeros_like(img)
    
    # Image centers
    center_x = width / 2
    center_y = height / 2
    
    # Apply rotation to each pixel
    for y in range(height):
        for x in range(width):
            # Translate to origin
            rel_x = x - center_x
            rel_y = y - center_y
            
            # Apply inverse rotation to find source pixel
            orig_x = cos_a * rel_x + sin_a * rel_y + center_x
            orig_y = -sin_a * rel_x + cos_a * rel_y + center_y
            
            # Round to nearest pixel
            orig_x = int(round(orig_x))
            orig_y = int(round(orig_y))
            
            # Check bounds and copy pixel
            if 0 <= orig_x < width and 0 <= orig_y < height:
                output[y, x] = img[orig_y, orig_x]
    
    return output

def scale_image(img, scale_x, scale_y):
    """Scale image by given factors"""
    height, width = img.shape[:2]
    
    # New dimensions
    new_height = int(height * scale_y)
    new_width = int(width * scale_x)
    
    # Create output image
    if len(img.shape) == 3:
        output = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)
    else:
        output = np.zeros((new_height, new_width), dtype=img.dtype)
    
    # Scale each pixel
    for y in range(new_height):
        for x in range(new_width):
            # Find corresponding pixel in original image
            orig_x = int(x / scale_x)
            orig_y = int(y / scale_y)
            
            # Check bounds and copy pixel
            if 0 <= orig_x < width and 0 <= orig_y < height:
                output[y, x] = img[orig_y, orig_x]
    
    return output

def translate_image(img, shift_x, shift_y):
    """Move image by given offsets"""
    height, width = img.shape[:2]
    
    # Create output image
    if len(img.shape) == 3:
        output = np.zeros_like(img)
    else:
        output = np.zeros_like(img)
    
    # Translate each pixel
    for y in range(height):
        for x in range(width):
            # Calculate new position
            new_x = x - shift_x
            new_y = y - shift_y
            
            # Check bounds and copy pixel
            if 0 <= new_x < width and 0 <= new_y < height:
                output[y, x] = img[new_y, new_x]
    
    return output

def rgb_to_gray(img):
    """Convert color image to grayscale"""
    if len(img.shape) == 2:
        return img.astype(float)
    
    # Use standard RGB weights
    gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    return gray

def find_edges(img):
    """Find edges using Sobel operator"""
    gray = rgb_to_gray(img)
    height, width = gray.shape
    
    # Sobel kernels
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    
    # Output arrays
    edges_x = np.zeros_like(gray)
    edges_y = np.zeros_like(gray)
    
    # Apply convolution (avoiding borders)
    for y in range(1, height-1):
        for x in range(1, width-1):
            # Get 3x3 neighborhood
            patch = gray[y-1:y+2, x-1:x+2]
            
            # Apply Sobel kernels
            edges_x[y, x] = np.sum(patch * sobel_x)
            edges_y[y, x] = np.sum(patch * sobel_y)
    
    # Combine gradients
    edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
    
    return edge_magnitude, edges_x, edges_y

def show_results():
    """Display all transformations"""
    print("Image Transformer - Linear Algebra Demo")
    print("="*40)
    
    # Create test image
    original = create_test_image()
    
    # Apply transformations
    rotated = rotate_image(original, 45)
    scaled = scale_image(original, 1.5, 1.5)
    translated = translate_image(original, 20, 10)
    
    # Edge detection
    edges, grad_x, grad_y = find_edges(original)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    # Show results
    axes[0,0].imshow(original)
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(rotated)
    axes[0,1].set_title('Rotated 45°')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(scaled)
    axes[0,2].set_title('Scaled 1.5x')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(translated)
    axes[0,3].set_title('Translated')
    axes[0,3].axis('off')
    
    axes[1,0].imshow(edges, cmap='gray')
    axes[1,0].set_title('Edge Detection')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(grad_x, cmap='gray')
    axes[1,1].set_title('X Gradient')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(grad_y, cmap='gray')
    axes[1,2].set_title('Y Gradient')
    axes[1,2].axis('off')
    
    # Show combined transformation
    combined = rotate_image(scaled, 30)
    axes[1,3].imshow(combined)
    axes[1,3].set_title('Combined')
    axes[1,3].axis('off')
    
    plt.tight_layout()
    plt.savefig('transformations.png', dpi=150)
    plt.show()
    
    print("All transformations complete!")
    print("Results saved as 'transformations.png'")

def show_math():
    """Demonstrate the math behind transformations"""
    print("\nMath Behind the Transformations:")
    print("-" * 30)
    
    # Rotation matrix example
    angle = 45
    rad = math.radians(angle)
    
    print(f"Rotation Matrix ({angle}°):")
    print(f"[ {math.cos(rad):.3f}  {-math.sin(rad):.3f} ]")
    print(f"[ {math.sin(rad):.3f}   {math.cos(rad):.3f} ]")
    print()
    
    # Test rotation on a point
    point = [1, 0]
    rotated_x = math.cos(rad) * point[0] - math.sin(rad) * point[1]
    rotated_y = math.sin(rad) * point[0] + math.cos(rad) * point[1]
    
    print(f"Point {point} rotated: [{rotated_x:.3f}, {rotated_y:.3f}]")
    print()
    
    # Scaling matrix
    print("Scaling Matrix (2x in X, 0.5x in Y):")
    print("[ 2.0  0.0 ]")
    print("[ 0.0  0.5 ]")
    print()
    
    # Edge detection explanation
    print("Edge Detection using Sobel:")
    print("Sobel X kernel finds vertical edges")
    print("[-1  0  1]")
    print("[-2  0  2]") 
    print("[-1  0  1]")

if __name__ == "__main__":
    show_math()
    show_results()
    print("\nProject 1 Complete! 🎉")
