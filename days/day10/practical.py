# Day 10: Normal Distribution
# The bell curve - nature's favorite shape
# Amey Prakash Sawant

print("Day 10: Normal Distribution")
print("=" * 27)

# Normal distribution = bell curve
# Most values cluster around the middle (mean)
# Few values at the extremes

import math

def normal_probability(x, mean, std_dev):
    # The famous bell curve formula
    # f(x) = (1 / (σ√2π)) × e^(-½((x-μ)/σ)²)
    
    # Step by step calculation
    pi = 3.14159
    e = 2.71828
    
    # Coefficient: 1 / (σ√2π)
    coefficient = 1 / (std_dev * (2 * pi) ** 0.5)
    
    # Exponent: -½((x-μ)/σ)²
    z_score = (x - mean) / std_dev
    exponent = -0.5 * z_score * z_score
    
    # Put it together
    probability = coefficient * (e ** exponent)
    return probability

# Example: Heights of people
print("\nExample: Adult male heights (inches)")
mean_height = 69  # 69 inches average
std_height = 3    # 3 inches standard deviation

heights = [63, 66, 69, 72, 75]  # Various heights
print(f"Mean height: {mean_height} inches")
print(f"Standard deviation: {std_height} inches")
print("\nHeight → Probability density:")

for height in heights:
    prob = normal_probability(height, mean_height, std_height)
    z = (height - mean_height) / std_height
    print(f"{height}" → {prob:.4f} (z-score: {z:.1f})")
# The 68-95-99.7 rule (empirical rule)
print(f"\nThe 68-95-99.7 Rule:")
print(f"For any normal distribution:")
print(f"• 68% of data within 1 standard deviation")
print(f"• 95% of data within 2 standard deviations") 
print(f"• 99.7% of data within 3 standard deviations")

print(f"\nFor our height example:")
range_1sd = f"{mean_height - std_height} to {mean_height + std_height} inches"
range_2sd = f"{mean_height - 2*std_height} to {mean_height + 2*std_height} inches"
range_3sd = f"{mean_height - 3*std_height} to {mean_height + 3*std_height} inches"

print(f"• 68% of men: {range_1sd}")
print(f"• 95% of men: {range_2sd}")
print(f"• 99.7% of men: {range_3sd}")

# Z-scores - standardized values
print(f"\nZ-scores (standard scores):")
print(f"z = (x - mean) / standard_deviation")
print(f"\nTells you how many standard deviations from mean")

for height in [72, 75, 78]:
    z = (height - mean_height) / std_height
    if z > 0:
        direction = "above"
    else:
        direction = "below"
    print(f"{height} inches → z = {z:.1f} ({abs(z):.1f} std devs {direction} average)")

print(f"\nWhy normal distribution matters:")
print(f"• Shows up everywhere in nature")
print(f"• Heights, test scores, measurement errors")
print(f"• Central Limit Theorem - averages become normal")
print(f"• Foundation of many statistical tests")

print("\nDay 10 complete! ✅")
    ax1.set_xlim(-8, 8)
    
    # Different standard deviations
    ax2 = axes[0, 1]
    for sigma in [0.5, 1, 2]:
        y = normal_pdf(x, 0, sigma)
        ax2.plot(x, y, label=f'μ=0, σ={sigma}', linewidth=2)
    ax2.set_title('Effect of Standard Deviation (σ)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-8, 8)
    
    # 68-95-99.7 Rule
    ax3 = axes[1, 0]
    x_rule = np.linspace(-4, 4, 1000)
    y_rule = normal_pdf(x_rule, 0, 1)
    
    ax3.plot(x_rule, y_rule, 'b-', linewidth=2)
    ax3.fill_between(x_rule, y_rule, where=(np.abs(x_rule) <= 1), 
                     alpha=0.3, color='green', label='68% (±1σ)')
    ax3.fill_between(x_rule, y_rule, where=(np.abs(x_rule) <= 2) & (np.abs(x_rule) > 1), 
                     alpha=0.3, color='yellow', label='95% (±2σ)')
    ax3.fill_between(x_rule, y_rule, where=(np.abs(x_rule) <= 3) & (np.abs(x_rule) > 2), 
                     alpha=0.3, color='red', label='99.7% (±3σ)')
    
    ax3.set_title('68-95-99.7 Rule')
    ax3.set_xlabel('Standard Deviations from Mean')
    ax3.set_ylabel('f(x)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # CDF
    ax4 = axes[1, 1]
    cdf_values = normal_cdf(x, 0, 1)
    ax4.plot(x, cdf_values, 'b-', linewidth=2)
    
    # Mark important points
    for p in [0.025, 0.5, 0.975]:
        z = stats.norm.ppf(p)
        ax4.axhline(y=p, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=z, color='gray', linestyle='--', alpha=0.5)
        ax4.plot(z, p, 'ro', markersize=8)
        ax4.annotate(f'({z:.2f}, {p})', xy=(z, p), xytext=(z+0.5, p+0.05))
    
    ax4.set_title('Cumulative Distribution Function (CDF)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('P(X ≤ x)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-4, 4)
    
    plt.tight_layout()
    plt.savefig('normal_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_clt():
    """Demonstrate Central Limit Theorem."""
    np.random.seed(42)
    
    # Original distribution (uniform)
    population = np.random.uniform(0, 10, 100000)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Original population
    ax = axes[0, 0]
    ax.hist(population, bins=50, density=True, alpha=0.7, color='blue')
    ax.set_title(f'Population (Uniform)\nμ={np.mean(population):.2f}, σ={np.std(population):.2f}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    
    # Sample means for different sample sizes
    sample_sizes = [1, 5, 30, 100, 500]
    
    for idx, n in enumerate(sample_sizes):
        ax = axes[(idx+1) // 3, (idx+1) % 3]
        
        # Take 1000 samples of size n and compute means
        sample_means = [np.mean(np.random.choice(population, n)) for _ in range(1000)]
        
        ax.hist(sample_means, bins=50, density=True, alpha=0.7, color='green')
        
        # Overlay theoretical normal
        x = np.linspace(min(sample_means), max(sample_means), 100)
        theoretical_std = np.std(population) / np.sqrt(n)
        y = normal_pdf(x, np.mean(population), theoretical_std)
        ax.plot(x, y, 'r-', linewidth=2, label='Theoretical Normal')
        
        ax.set_title(f'Sample Size n={n}\nμ={np.mean(sample_means):.2f}, σ={np.std(sample_means):.2f}')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.suptitle('Central Limit Theorem Demonstration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('central_limit_theorem.png', dpi=150, bbox_inches='tight')
    plt.show()


def probability_calculations():
    """Demonstrate probability calculations with normal distribution."""
    mu, sigma = 100, 15  # IQ distribution
    
    print("=" * 60)
    print("Normal Distribution Probability Calculations")
    print("=" * 60)
    print(f"\nIQ Distribution: μ = {mu}, σ = {sigma}")
    
    # P(X < 115)
    prob1 = stats.norm.cdf(115, mu, sigma)
    print(f"\nP(IQ < 115) = {prob1:.4f} ({prob1*100:.2f}%)")
    
    # P(X > 130)
    prob2 = 1 - stats.norm.cdf(130, mu, sigma)
    print(f"P(IQ > 130) = {prob2:.4f} ({prob2*100:.2f}%)")
    
    # P(85 < X < 115)
    prob3 = stats.norm.cdf(115, mu, sigma) - stats.norm.cdf(85, mu, sigma)
    print(f"P(85 < IQ < 115) = {prob3:.4f} ({prob3*100:.2f}%)")
    
    # Find z-score
    z = z_score(130, mu, sigma)
    print(f"\nZ-score for IQ=130: {z:.2f}")
    
    # Find percentile (inverse CDF)
    percentile_90 = stats.norm.ppf(0.90, mu, sigma)
    print(f"90th percentile IQ: {percentile_90:.2f}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.linspace(50, 150, 1000)
    y = normal_pdf(x, mu, sigma)
    
    ax.plot(x, y, 'b-', linewidth=2)
    
    # Shade P(X < 115)
    x_fill = x[x <= 115]
    y_fill = normal_pdf(x_fill, mu, sigma)
    ax.fill_between(x_fill, y_fill, alpha=0.3, color='green', 
                    label=f'P(X < 115) = {prob1:.2%}')
    
    # Shade P(X > 130)
    x_fill2 = x[x >= 130]
    y_fill2 = normal_pdf(x_fill2, mu, sigma)
    ax.fill_between(x_fill2, y_fill2, alpha=0.3, color='red',
                    label=f'P(X > 130) = {prob2:.2%}')
    
    ax.axvline(x=mu, color='black', linestyle='--', label=f'Mean = {mu}')
    ax.set_xlabel('IQ Score')
    ax.set_ylabel('Probability Density')
    ax.set_title('IQ Distribution (Normal: μ=100, σ=15)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('probability_calculations.png', dpi=150, bbox_inches='tight')
    plt.show()


def normality_test():
    """Demonstrate normality testing."""
    np.random.seed(42)
    
    # Generate samples
    normal_data = np.random.normal(0, 1, 500)
    exponential_data = np.random.exponential(1, 500)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Normal data histogram
    ax1 = axes[0, 0]
    ax1.hist(normal_data, bins=30, density=True, alpha=0.7, color='blue')
    x = np.linspace(-4, 4, 100)
    ax1.plot(x, normal_pdf(x, np.mean(normal_data), np.std(normal_data)), 
             'r-', linewidth=2, label='Fitted Normal')
    ax1.set_title('Normal Data')
    ax1.legend()
    
    # Normal Q-Q plot
    ax2 = axes[0, 1]
    stats.probplot(normal_data, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal Data)')
    
    # Exponential data histogram
    ax3 = axes[1, 0]
    ax3.hist(exponential_data, bins=30, density=True, alpha=0.7, color='orange')
    ax3.set_title('Exponential Data (Non-Normal)')
    
    # Exponential Q-Q plot
    ax4 = axes[1, 1]
    stats.probplot(exponential_data, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Exponential Data)')
    
    plt.tight_layout()
    plt.savefig('normality_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistical tests for normality
    print("\n" + "=" * 60)
    print("Normality Tests")
    print("=" * 60)
    
    # Shapiro-Wilk test
    stat1, p1 = stats.shapiro(normal_data)
    stat2, p2 = stats.shapiro(exponential_data)
    
    print("\nShapiro-Wilk Test:")
    print(f"  Normal data:      statistic={stat1:.4f}, p-value={p1:.4f}")
    print(f"  Exponential data: statistic={stat2:.4f}, p-value={p2:.4f}")
    
    print("\nInterpretation (α=0.05):")
    print(f"  Normal data: {'Normal' if p1 > 0.05 else 'Not normal'}")
    print(f"  Exponential data: {'Normal' if p2 > 0.05 else 'Not normal'}")


if __name__ == "__main__":
    print("=" * 60)
    print("Day 10: Normal Distribution")
    print("=" * 60)
    
    # Visualize normal distribution
    visualize_normal_distribution()
    
    # Demonstrate CLT
    demonstrate_clt()
    
    # Probability calculations
    probability_calculations()
    
    # Normality testing
    normality_test()
    
    print("\n✅ Day 10 Complete!")
