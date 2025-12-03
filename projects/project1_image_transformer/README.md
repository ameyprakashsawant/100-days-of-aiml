# Project 1: Image Transformer (Pure Linear Algebra)

A project implementing image transformations using pure linear algebra concepts.

## 📚 Concepts Used

- Matrix Multiplication
- Rotation Matrices
- Dot Products
- Norms
- Convolution (as matrix operation)

## 🎯 Project Goals

1. **Rotate Image** using rotation matrix
2. **Scale and Translate** images
3. **Detect Edges** using convolution

## 📁 Project Structure

```
project1_image_transformer/
├── README.md
├── image_transformer.py
├── edge_detection.py
├── examples/
│   └── sample_images/
└── requirements.txt
```

## 🚀 How to Run

```bash
pip install -r requirements.txt
python image_transformer.py
```

## 📖 Theory Behind the Code

### 1. Rotation Matrix

A 2D rotation by angle θ:

```
R(θ) = | cos(θ)  -sin(θ) |
       | sin(θ)   cos(θ) |
```

For each pixel (x, y), new position = R(θ) × [x, y]ᵀ

### 2. Scaling Matrix

```
S = | sx  0 |
    | 0  sy |
```

### 3. Translation

```
[x', y']ᵀ = [x, y]ᵀ + [tx, ty]ᵀ
```

Using homogeneous coordinates for combined transforms:

```
| x' |   | sx*cos(θ)  -sy*sin(θ)  tx |   | x |
| y' | = | sx*sin(θ)   sy*cos(θ)  ty | × | y |
| 1  |   |    0           0        1 |   | 1 |
```

### 4. Edge Detection (Convolution)

Sobel operator:

```
Gx = | -1  0  1 |      Gy = | -1 -2 -1 |
     | -2  0  2 |           |  0  0  0 |
     | -1  0  1 |           |  1  2  1 |
```

Edge magnitude: √(Gx² + Gy²) (using norms!)
Edge direction: arctan(Gy/Gx) (using dot products!)
