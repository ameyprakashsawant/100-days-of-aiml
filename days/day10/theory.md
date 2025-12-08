# Day 10: Normal Distribution

## 📚 Theory

### What is Normal Distribution?

The **normal distribution** (Gaussian distribution) is the most important probability distribution in statistics. It's characterized by:

- Bell-shaped, symmetric curve
- Mean = Median = Mode
- Defined by two parameters: μ (mean) and σ (standard deviation)

### Probability Density Function

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

### Standard Normal Distribution

When μ = 0 and σ = 1:
$$Z = \frac{X - \mu}{\sigma}$$

### The 68-95-99.7 Rule

- **68%** of data falls within 1σ of the mean
- **95%** of data falls within 2σ of the mean
- **99.7%** of data falls within 3σ of the mean

### Why It's Important in ML

1. **Central Limit Theorem**: Sample means follow normal distribution
2. **Feature normalization**: Many algorithms assume normally distributed features
3. **Error distribution**: Residuals often assumed to be normal
4. **Statistical tests**: Many tests assume normality

### Z-Score

The z-score tells us how many standard deviations away from the mean a value is:
$$z = \frac{x - \mu}{\sigma}$$

### Cumulative Distribution Function (CDF)

$$\Phi(x) = P(X \leq x) = \int_{-\infty}^{x} f(t) dt$$

## 🔑 Key Formulas

| Concept            | Formula                            |
| ------------------ | ---------------------------------- |
| Mean               | μ                                  |
| Variance           | σ²                                 |
| Standard Deviation | σ                                  |
| Z-Score            | z = (x - μ) / σ                    |
| PDF                | f(x) = (1/σ√2π) × e^(-½((x-μ)/σ)²) |

## 🎯 ML Applications

- **Gaussian Naive Bayes**: Assumes features follow normal distribution
- **Anomaly Detection**: Identify outliers beyond 3σ
- **Confidence Intervals**: Statistical inference
- **Regularization**: L2 regularization equivalent to Gaussian prior
