# Day 8: Mean, Variance & Standard Deviation
# Understanding how spread out data is
# Amey Prakash Sawant

print("Day 8: Mean, Variance & Standard Deviation")
print("=" * 40)

# Working with some test scores
data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 100]
print(f"Test scores: {data}")

# Mean = average (add all up, divide by count)
total = 0
for score in data:
    total += score

mean = total / len(data)
print(f"\nMean calculation:")
print(f"Sum = {total}")
print(f"Count = {len(data)}")
print(f"Mean = {total}/{len(data)} = {mean}")

# The outlier (100) really affects the mean!
data_no_outlier = data[:-1]  # Remove the 100
total_no_outlier = 0
for score in data_no_outlier:
    total_no_outlier += score
mean_no_outlier = total_no_outlier / len(data_no_outlier)

print(f"\nWithout the outlier (100):")
print(f"Mean = {mean_no_outlier}")
print(f"With outlier: Mean = {mean}")
print("→ Outliers pull the mean towards them!")

# Variance = how spread out the data is
print(f"\nVariance calculation:")
print(f"Step 1: Find differences from mean")

differences = []
for score in data:
    diff = score - mean
    differences.append(diff)
    print(f"  {score} - {mean} = {diff}")

print(f"\nStep 2: Square the differences")
squared_diffs = []
for diff in differences:
    squared = diff * diff
    squared_diffs.append(squared)
    print(f"  {diff}² = {squared}")

print(f"\nStep 3: Average the squared differences")
variance = sum(squared_diffs) / len(squared_diffs)
print(f"Variance = {sum(squared_diffs)}/{len(squared_diffs)} = {variance:.2f}")

# Standard deviation = square root of variance
std_dev = variance ** 0.5
print(f"\nStandard deviation = √{variance:.2f} = {std_dev:.2f}")

print(f"\nWhat this means:")
print(f"- Most scores are within {std_dev:.1f} points of the mean ({mean:.1f})")
print(f"- Range: {mean - std_dev:.1f} to {mean + std_dev:.1f}")

print("\nDay 8 complete! ✅")

print("\n" + "=" * 50)
print("2. MEDIAN")
print("=" * 50)

median = np.median(data)
print(f"Data: {data}")
print(f"Median: {median}")
print(f"Mean: {np.mean(data):.2f}")
print("→ Median (26.5) is more representative than mean (32.5)")

# Odd number of elements
data_odd = np.array([1, 3, 5, 7, 9])
print(f"\nOdd data: {data_odd}")
print(f"Median: {np.median(data_odd)} (middle element)")


# ============================================
# 3. MODE
# ============================================

print("\n" + "=" * 50)
print("3. MODE")
print("=" * 50)

data_mode = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
mode_result = stats.mode(data_mode, keepdims=True)
print(f"Data: {data_mode}")
print(f"Mode: {mode_result.mode[0]} (appears {mode_result.count[0]} times)")

# Bimodal distribution
data_bimodal = np.array([1, 1, 1, 5, 5, 5, 3])
print(f"\nBimodal data: {data_bimodal}")
values, counts = np.unique(data_bimodal, return_counts=True)
modes = values[counts == counts.max()]
print(f"Modes: {modes}")


# ============================================
# 4. VARIANCE
# ============================================

print("\n" + "=" * 50)
print("4. VARIANCE")
print("=" * 50)

data = np.array([2, 4, 4, 4, 5, 5, 7, 9])

# Population variance (ddof=0)
var_pop = np.var(data, ddof=0)
# Sample variance (ddof=1) - Bessel's correction
var_sample = np.var(data, ddof=1)

# Manual calculation
mean = np.mean(data)
squared_diffs = (data - mean) ** 2
var_manual = np.sum(squared_diffs) / len(data)

print(f"Data: {data}")
print(f"Mean: {mean}")
print(f"Squared differences from mean: {squared_diffs}")
print(f"\nPopulation variance (÷n): {var_pop:.4f}")
print(f"Sample variance (÷(n-1)): {var_sample:.4f}")
print(f"Manual (population): {var_manual:.4f}")


# ============================================
# 5. STANDARD DEVIATION
# ============================================

print("\n" + "=" * 50)
print("5. STANDARD DEVIATION")
print("=" * 50)

std_pop = np.std(data, ddof=0)
std_sample = np.std(data, ddof=1)

print(f"Population std dev: {std_pop:.4f}")
print(f"Sample std dev: {std_sample:.4f}")
print(f"Variance = std² : {std_pop**2:.4f} = {var_pop:.4f}")

# Coefficient of variation
cv = (std_sample / mean) * 100
print(f"\nCoefficient of Variation: {cv:.2f}%")


# ============================================
# 6. Z-SCORE (STANDARDIZATION)
# ============================================

print("\n" + "=" * 50)
print("6. Z-SCORE (STANDARDIZATION)")
print("=" * 50)

def z_score(x, mean, std):
    return (x - mean) / std

# IQ scores (mean=100, std=15)
iq_scores = np.array([85, 100, 115, 130, 145])
iq_mean, iq_std = 100, 15

print("IQ Score → Z-Score:")
for iq in iq_scores:
    z = z_score(iq, iq_mean, iq_std)
    print(f"  IQ {iq} → z = {z:+.2f}")

# Standardize data
data_std = (data - np.mean(data)) / np.std(data)
print(f"\nOriginal data: {data}")
print(f"Standardized: {data_std.round(2)}")
print(f"New mean: {np.mean(data_std):.4f}")
print(f"New std: {np.std(data_std):.4f}")


# ============================================
# 7. EMPIRICAL RULE (68-95-99.7)
# ============================================

print("\n" + "=" * 50)
print("7. EMPIRICAL RULE (68-95-99.7)")
print("=" * 50)

# Generate normal data
np.random.seed(42)
normal_data = np.random.normal(loc=100, scale=15, size=10000)

mean = np.mean(normal_data)
std = np.std(normal_data)

within_1_std = np.sum(np.abs(normal_data - mean) <= std) / len(normal_data) * 100
within_2_std = np.sum(np.abs(normal_data - mean) <= 2*std) / len(normal_data) * 100
within_3_std = np.sum(np.abs(normal_data - mean) <= 3*std) / len(normal_data) * 100

print(f"Normal data: mean={mean:.2f}, std={std:.2f}")
print(f"\nWithin 1σ: {within_1_std:.1f}% (expected: 68%)")
print(f"Within 2σ: {within_2_std:.1f}% (expected: 95%)")
print(f"Within 3σ: {within_3_std:.1f}% (expected: 99.7%)")


# ============================================
# 8. ROBUST STATISTICS
# ============================================

print("\n" + "=" * 50)
print("8. ROBUST STATISTICS")
print("=" * 50)

data_outliers = np.array([10, 12, 14, 15, 16, 18, 20, 200])

# Standard statistics
mean = np.mean(data_outliers)
std = np.std(data_outliers)

# Robust statistics
median = np.median(data_outliers)
q1, q3 = np.percentile(data_outliers, [25, 75])
iqr = q3 - q1
mad = np.median(np.abs(data_outliers - median))

print(f"Data: {data_outliers}")
print(f"\nStandard statistics:")
print(f"  Mean: {mean:.2f}")
print(f"  Std: {std:.2f}")
print(f"\nRobust statistics:")
print(f"  Median: {median:.2f}")
print(f"  IQR: {iqr:.2f}")
print(f"  MAD: {mad:.2f}")

# Outlier detection using IQR
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data_outliers[(data_outliers < lower_bound) | (data_outliers > upper_bound)]
print(f"\nOutlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Detected outliers: {outliers}")


# ============================================
# 9. ML APPLICATION: Feature Scaling
# ============================================

print("\n" + "=" * 50)
print("9. ML APPLICATION: Feature Scaling")
print("=" * 50)

# Sample feature data
np.random.seed(42)
ages = np.random.randint(20, 60, size=10)
salaries = np.random.randint(30000, 150000, size=10)

print(f"Ages: {ages}")
print(f"  Mean: {np.mean(ages):.2f}, Std: {np.std(ages):.2f}")
print(f"\nSalaries: {salaries}")
print(f"  Mean: {np.mean(salaries):.2f}, Std: {np.std(salaries):.2f}")

# Standardization (Z-score)
ages_scaled = (ages - np.mean(ages)) / np.std(ages)
salaries_scaled = (salaries - np.mean(salaries)) / np.std(salaries)

print(f"\nAfter Standardization:")
print(f"Ages scaled: {ages_scaled.round(2)}")
print(f"Salaries scaled: {salaries_scaled.round(2)}")
print(f"\nNow both have mean≈0, std≈1 - ready for ML!")

# Min-Max normalization
ages_minmax = (ages - ages.min()) / (ages.max() - ages.min())
print(f"\nMin-Max normalized ages: {ages_minmax.round(2)}")
print(f"Range: [{ages_minmax.min()}, {ages_minmax.max()}]")


# ============================================
# 10. VISUALIZATION
# ============================================

print("\n" + "=" * 50)
print("10. VISUALIZATION")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Distribution with mean, median, mode
ax1 = axes[0, 0]
skewed_data = np.random.exponential(scale=2, size=1000)
ax1.hist(skewed_data, bins=30, density=True, alpha=0.7, color='blue')
ax1.axvline(np.mean(skewed_data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(skewed_data):.2f}')
ax1.axvline(np.median(skewed_data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(skewed_data):.2f}')
ax1.legend()
ax1.set_title('Skewed Distribution: Mean vs Median')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')

# Plot 2: Empirical Rule
ax2 = axes[0, 1]
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)
ax2.plot(x, y, 'b-', linewidth=2)
ax2.fill_between(x, y, where=(np.abs(x) <= 1), alpha=0.3, color='green', label='68% (1σ)')
ax2.fill_between(x, y, where=(np.abs(x) <= 2) & (np.abs(x) > 1), alpha=0.3, color='yellow', label='95% (2σ)')
ax2.fill_between(x, y, where=(np.abs(x) <= 3) & (np.abs(x) > 2), alpha=0.3, color='red', label='99.7% (3σ)')
ax2.legend()
ax2.set_title('Empirical Rule (68-95-99.7)')
ax2.set_xlabel('Standard Deviations')
ax2.set_ylabel('Density')

# Plot 3: Effect of outliers
ax3 = axes[1, 0]
clean_data = np.random.normal(50, 10, 100)
outlier_data = np.append(clean_data, [150, 160, 170])

ax3.boxplot([clean_data, outlier_data], labels=['Clean', 'With Outliers'])
ax3.scatter([2]*3, [150, 160, 170], color='red', s=50, zorder=5, label='Outliers')
ax3.legend()
ax3.set_title('Effect of Outliers on Distribution')
ax3.set_ylabel('Value')

# Plot 4: Before and After Standardization
ax4 = axes[1, 1]
feature1 = np.random.normal(100, 10, 100)
feature2 = np.random.normal(0.5, 0.1, 100)

ax4.scatter(feature1, feature2, alpha=0.5, label='Original')

# Standardize both
f1_std = (feature1 - np.mean(feature1)) / np.std(feature1)
f2_std = (feature2 - np.mean(feature2)) / np.std(feature2)

ax4.scatter(f1_std, f2_std, alpha=0.5, label='Standardized')
ax4.legend()
ax4.set_title('Before and After Standardization')
ax4.set_xlabel('Feature 1')
ax4.set_ylabel('Feature 2')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('statistics_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'statistics_visualization.png'")


print("\n" + "=" * 50)
print("Day 8 Complete! 🎉")
print("=" * 50)
