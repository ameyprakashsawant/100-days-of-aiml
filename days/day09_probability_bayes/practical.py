# Day 9: Probability & Bayes' Theorem
# The math of uncertainty and updating beliefs
# Amey Prakash Sawant

print("Day 9: Probability & Bayes' Theorem")
print("=" * 35)

# Basic probability with coin flips
import random

# Simulate coin flips the simple way
heads_count = 0
total_flips = 1000

print(f"Flipping coin {total_flips} times...")
for i in range(total_flips):
    flip = random.choice(['H', 'T'])
    if flip == 'H':
        heads_count += 1

probability_heads = heads_count / total_flips
print(f"\nHeads: {heads_count} out of {total_flips}")
print(f"P(Heads) = {heads_count}/{total_flips} = {probability_heads:.3f}")
print(f"Expected: 0.500 (should be close!)")


# Conditional probability - probability given some info
print(f"\nConditional Probability:")
print("Example: What's P(rain | cloudy)?")
print("= Probability of rain GIVEN it's cloudy")

# Simple weather example
cloudy_days = 100
rainy_and_cloudy = 60
rainy_given_cloudy = rainy_and_cloudy / cloudy_days

print(f"\nOut of {cloudy_days} cloudy days, {rainy_and_cloudy} had rain")
print(f"P(rain | cloudy) = {rainy_and_cloudy}/{cloudy_days} = {rainy_given_cloudy}")

# Bayes' Theorem - updating beliefs with new evidence
print(f"\nBayes' Theorem:")
print("P(A|B) = P(B|A) × P(A) / P(B)")
print("\nMedical test example:")

# Disease probability
p_disease = 0.001  # 0.1% of people have disease
p_no_disease = 1 - p_disease

# Test accuracy
p_positive_given_disease = 0.99     # 99% chance positive if you have disease
p_positive_given_no_disease = 0.05   # 5% false positive rate

# Total probability of positive test
p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_no_disease * p_no_disease)

# What we really want: P(disease | positive test)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"Prior: P(disease) = {p_disease*100}%")
print(f"Test accuracy: {p_positive_given_disease*100}%")
print(f"False positive rate: {p_positive_given_no_disease*100}%")
print(f"\nAfter positive test:")
print(f"P(disease | positive) = {p_disease_given_positive:.3f} = {p_disease_given_positive*100:.1f}%")
print("\nSurprise! Even with 99% accurate test, only ~2% chance you have disease!")
print("This is because the disease is very rare to begin with.")

print("\nDay 9 complete! ✅")

# P(sum > 8 | die1 = 5)
condition = die1 == 5
p_sum_gt_8_given_die1_5 = np.sum((total > 8) & condition) / np.sum(condition)

print("Two dice rolled:")
print(f"P(sum > 8 | die1 = 5) = {p_sum_gt_8_given_die1_5:.4f}")
print(f"Theoretical: P(die2 >= 4) = 3/6 = 0.5")


# ============================================
# 3. BAYES' THEOREM
# ============================================

print("\n" + "=" * 50)
print("3. BAYES' THEOREM")
print("=" * 50)

def bayes_theorem(p_b_given_a, p_a, p_b):
    """Calculate P(A|B) using Bayes' theorem"""
    return (p_b_given_a * p_a) / p_b

# Medical test example
p_disease = 0.01  # 1% of population has disease
p_positive_given_disease = 0.95  # Test sensitivity
p_positive_given_no_disease = 0.05  # False positive rate

# P(Positive) using total probability
p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_no_disease * (1 - p_disease))

# P(Disease | Positive) using Bayes
p_disease_given_positive = bayes_theorem(
    p_positive_given_disease, p_disease, p_positive
)

print("Medical Test Example:")
print(f"P(Disease) = {p_disease:.2%}")
print(f"P(Positive | Disease) = {p_positive_given_disease:.2%}")
print(f"P(Positive | No Disease) = {p_positive_given_no_disease:.2%}")
print(f"\nP(Positive) = {p_positive:.4f}")
print(f"P(Disease | Positive) = {p_disease_given_positive:.4f}")
print(f"\n→ Only {p_disease_given_positive:.1%} chance of disease given positive test!")
print("→ This is the 'Base Rate Fallacy' - rare diseases are still unlikely!")


# ============================================
# 4. SIMULATION OF BAYES
# ============================================

print("\n" + "=" * 50)
print("4. SIMULATION OF BAYES")
print("=" * 50)

np.random.seed(42)
population = 100000

# Generate population
has_disease = np.random.random(population) < p_disease

# Generate test results
test_result = np.zeros(population, dtype=bool)
test_result[has_disease] = np.random.random(np.sum(has_disease)) < p_positive_given_disease
test_result[~has_disease] = np.random.random(np.sum(~has_disease)) < p_positive_given_no_disease

# Calculate from simulation
positive_tests = np.sum(test_result)
true_positives = np.sum(test_result & has_disease)
p_disease_given_positive_sim = true_positives / positive_tests

print(f"Simulation with {population:,} people:")
print(f"People with disease: {np.sum(has_disease):,}")
print(f"Positive tests: {positive_tests:,}")
print(f"True positives: {true_positives:,}")
print(f"P(Disease | Positive) simulated: {p_disease_given_positive_sim:.4f}")
print(f"P(Disease | Positive) theoretical: {p_disease_given_positive:.4f}")


# ============================================
# 5. NAIVE BAYES CLASSIFIER
# ============================================

print("\n" + "=" * 50)
print("5. NAIVE BAYES CLASSIFIER (Simple Example)")
print("=" * 50)

# Spam classification example
# Features: contains "free", contains "money"
# Classes: spam, not spam

# Training data statistics (fictional)
p_spam = 0.3
p_free_given_spam = 0.8
p_free_given_not_spam = 0.1
p_money_given_spam = 0.7
p_money_given_not_spam = 0.05

def naive_bayes_classify(contains_free, contains_money):
    """Classify email as spam or not using Naive Bayes"""
    # P(spam | features) ∝ P(spam) * P(features | spam)
    # Using conditional independence: P(features | spam) = P(free|spam) * P(money|spam)
    
    # Likelihood for spam
    p_features_spam = 1.0
    if contains_free:
        p_features_spam *= p_free_given_spam
    else:
        p_features_spam *= (1 - p_free_given_spam)
    if contains_money:
        p_features_spam *= p_money_given_spam
    else:
        p_features_spam *= (1 - p_money_given_spam)
    
    # Likelihood for not spam
    p_features_not_spam = 1.0
    if contains_free:
        p_features_not_spam *= p_free_given_not_spam
    else:
        p_features_not_spam *= (1 - p_free_given_not_spam)
    if contains_money:
        p_features_not_spam *= p_money_given_not_spam
    else:
        p_features_not_spam *= (1 - p_money_given_not_spam)
    
    # Unnormalized posteriors
    posterior_spam = p_spam * p_features_spam
    posterior_not_spam = (1 - p_spam) * p_features_not_spam
    
    # Normalize
    total = posterior_spam + posterior_not_spam
    p_spam_given_features = posterior_spam / total
    
    return p_spam_given_features

# Test cases
test_cases = [
    (True, True, "Contains 'free' and 'money'"),
    (True, False, "Contains only 'free'"),
    (False, True, "Contains only 'money'"),
    (False, False, "Contains neither"),
]

print("Email Spam Classification:")
for free, money, desc in test_cases:
    p = naive_bayes_classify(free, money)
    label = "SPAM" if p > 0.5 else "NOT SPAM"
    print(f"  {desc}: P(spam) = {p:.4f} → {label}")


# ============================================
# 6. PROBABILITY DISTRIBUTIONS
# ============================================

print("\n" + "=" * 50)
print("6. PROBABILITY DISTRIBUTIONS")
print("=" * 50)

# Binomial: number of successes in n trials
n_trials = 10
p_success = 0.3
binomial_samples = np.random.binomial(n_trials, p_success, size=10000)

print(f"Binomial(n={n_trials}, p={p_success}):")
print(f"  Mean: {np.mean(binomial_samples):.2f} (theoretical: {n_trials * p_success})")
print(f"  Std: {np.std(binomial_samples):.2f}")

# Poisson: count of rare events
lambda_param = 5
poisson_samples = np.random.poisson(lambda_param, size=10000)

print(f"\nPoisson(λ={lambda_param}):")
print(f"  Mean: {np.mean(poisson_samples):.2f} (theoretical: {lambda_param})")
print(f"  Std: {np.std(poisson_samples):.2f}")


# ============================================
# 7. VISUALIZATION
# ============================================

print("\n" + "=" * 50)
print("7. VISUALIZATION")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Medical Test Bayes Visualization
ax1 = axes[0, 0]
categories = ['Has Disease\nTest +', 'Has Disease\nTest -', 
              'No Disease\nTest +', 'No Disease\nTest -']
# Using actual counts from simulation
counts = [
    true_positives,
    np.sum(has_disease & ~test_result),
    np.sum(~has_disease & test_result),
    np.sum(~has_disease & ~test_result)
]
colors = ['red', 'orange', 'yellow', 'green']
ax1.bar(categories, counts, color=colors)
ax1.set_title('Medical Test Results (Base Rate Fallacy)')
ax1.set_ylabel('Count')
for i, v in enumerate(counts):
    ax1.text(i, v + 500, f'{v:,}', ha='center')

# Plot 2: Prior vs Posterior
ax2 = axes[0, 1]
diseases = np.linspace(0.001, 0.1, 100)
posteriors = []
for p_d in diseases:
    p_pos = p_positive_given_disease * p_d + p_positive_given_no_disease * (1 - p_d)
    post = (p_positive_given_disease * p_d) / p_pos
    posteriors.append(post)

ax2.plot(diseases * 100, np.array(posteriors) * 100)
ax2.axhline(y=50, color='r', linestyle='--', label='50% threshold')
ax2.axvline(x=1, color='g', linestyle='--', label='Our example (1%)')
ax2.set_xlabel('Prior P(Disease) %')
ax2.set_ylabel('Posterior P(Disease|Positive) %')
ax2.set_title('How Base Rate Affects Posterior')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Binomial Distribution
ax3 = axes[1, 0]
k = np.arange(0, n_trials + 1)
pmf = stats.binom.pmf(k, n_trials, p_success)
ax3.bar(k, pmf, alpha=0.7)
ax3.axvline(x=n_trials * p_success, color='r', linestyle='--', label='Mean')
ax3.set_xlabel('Number of Successes')
ax3.set_ylabel('Probability')
ax3.set_title(f'Binomial(n={n_trials}, p={p_success})')
ax3.legend()

# Plot 4: Poisson Distribution
ax4 = axes[1, 1]
k = np.arange(0, 15)
pmf = stats.poisson.pmf(k, lambda_param)
ax4.bar(k, pmf, alpha=0.7)
ax4.axvline(x=lambda_param, color='r', linestyle='--', label='λ (mean)')
ax4.set_xlabel('k (count)')
ax4.set_ylabel('Probability')
ax4.set_title(f'Poisson(λ={lambda_param})')
ax4.legend()

plt.tight_layout()
plt.savefig('probability_bayes_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved as 'probability_bayes_visualization.png'")


print("\n" + "=" * 50)
print("Day 9 Complete! 🎉")
print("=" * 50)
