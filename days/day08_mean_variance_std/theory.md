# Day 8: Mean, Variance & Standard Deviation

## 📚 Learning Objectives

- Master central tendency measures
- Understand variance and standard deviation
- Apply these concepts to ML problems

---

## 1. Mean (Average)

### Population Mean

$$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$$

### Sample Mean

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

### Properties

- Sum of deviations from mean = 0
- Minimizes sum of squared errors
- Sensitive to outliers

### Weighted Mean

$$\bar{x}_w = \frac{\sum w_i x_i}{\sum w_i}$$

---

## 2. Median

### Definition

Middle value when data is sorted.

### Properties

- Robust to outliers
- For odd n: middle element
- For even n: average of two middle elements

### When to Use

- Skewed distributions
- Data with outliers (e.g., income data)

---

## 3. Mode

### Definition

Most frequently occurring value.

### Types

- **Unimodal**: One peak
- **Bimodal**: Two peaks
- **Multimodal**: Multiple peaks

---

## 4. Variance

### Population Variance

$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2$$

### Sample Variance (Bessel's correction)

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

### Why n-1?

- Sample mean is estimated from data
- Dividing by n underestimates true variance
- n-1 gives unbiased estimate

### Properties

- Always ≥ 0
- Variance of constant = 0
- Var(aX + b) = a²Var(X)

---

## 5. Standard Deviation

### Definition

$$\sigma = \sqrt{\text{Variance}}$$

### Properties

- Same units as original data
- SD(aX + b) = |a| × SD(X)
- More interpretable than variance

### Coefficient of Variation

$$CV = \frac{\sigma}{\mu} \times 100\%$$

Relative measure of spread.

---

## 6. Empirical Rule (68-95-99.7)

For normal distributions:

- **68%** of data within μ ± 1σ
- **95%** of data within μ ± 2σ
- **99.7%** of data within μ ± 3σ

### Z-Score (Standardization)

$$z = \frac{x - \mu}{\sigma}$$

Converts any normal distribution to standard normal N(0,1).

---

## 7. Applications in ML

### Feature Scaling

```python
# Standardization (Z-score normalization)
X_scaled = (X - X.mean()) / X.std()

# Min-Max normalization
X_norm = (X - X.min()) / (X.max() - X.min())
```

### Why Scale Features?

- Gradient descent converges faster
- Distance-based algorithms work better
- Regularization treats features fairly

### Loss Functions

- **MSE**: Uses squared deviations (like variance)
- **MAE**: Uses absolute deviations (more robust)

### Batch Statistics

- Batch Normalization uses mean/variance
- LayerNorm, InstanceNorm, GroupNorm

---

## 8. Robust Statistics

### Interquartile Range (IQR)

$$IQR = Q_3 - Q_1$$

Where Q1 = 25th percentile, Q3 = 75th percentile.

### Outlier Detection

Values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR are outliers.

### Median Absolute Deviation (MAD)

$$MAD = \text{median}(|x_i - \text{median}(x)|)$$

Robust alternative to standard deviation.

---

## 🔑 Key Takeaways

| Measure  | Formula      | Use Case           |
| -------- | ------------ | ------------------ |
| Mean     | Σx/n         | Symmetric data     |
| Median   | Middle value | Skewed data        |
| Variance | Σ(x-μ)²/n    | Spread measure     |
| Std Dev  | √Variance    | Same units as data |
| Z-score  | (x-μ)/σ      | Standardization    |

---

## 📖 Further Reading

- Statistics by Freedman, Pisani, Purves
- Think Stats by Allen Downey
