# Day 9: Probability Basics & Bayes' Theorem

## 📚 Learning Objectives

- Master fundamental probability concepts
- Understand conditional probability
- Apply Bayes' theorem to ML problems

---

## 1. Probability Basics

### Sample Space

Set of all possible outcomes: Ω

### Event

Subset of sample space: A ⊆ Ω

### Probability Axioms

1. P(A) ≥ 0 for all events
2. P(Ω) = 1
3. P(A ∪ B) = P(A) + P(B) if A ∩ B = ∅

---

## 2. Basic Rules

### Complement

$$P(A^c) = 1 - P(A)$$

### Addition Rule

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

### Multiplication Rule

$$P(A \cap B) = P(A) \cdot P(B|A)$$

---

## 3. Conditional Probability

### Definition

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

"Probability of A given B has occurred"

### Independence

Events A and B are independent if:
$$P(A|B) = P(A)$$
or equivalently:
$$P(A \cap B) = P(A) \cdot P(B)$$

---

## 4. Bayes' Theorem

### Formula

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

### Components

- **P(A)**: Prior probability
- **P(B|A)**: Likelihood
- **P(A|B)**: Posterior probability
- **P(B)**: Evidence (normalizing constant)

### Extended Form

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B|A) \cdot P(A) + P(B|A^c) \cdot P(A^c)}$$

---

## 5. Law of Total Probability

If B₁, B₂, ..., Bₙ partition Ω:
$$P(A) = \sum_{i=1}^{n} P(A|B_i) \cdot P(B_i)$$

---

## 6. Applications in ML

### Naive Bayes Classifier

$$P(y|x_1,...,x_n) \propto P(y) \prod_{i=1}^{n} P(x_i|y)$$

### Bayesian Inference

- Prior: What we believe before seeing data
- Likelihood: How probable is data given hypothesis
- Posterior: Updated belief after seeing data

### Medical Diagnosis Example

- P(Disease) = 0.01 (prior)
- P(Positive|Disease) = 0.95 (sensitivity)
- P(Positive|No Disease) = 0.05 (false positive)
- What's P(Disease|Positive)?

---

## 7. Common Probability Distributions

### Discrete

- **Bernoulli**: Single binary outcome
- **Binomial**: n independent Bernoulli trials
- **Poisson**: Count of rare events

### Continuous

- **Uniform**: Equal probability over interval
- **Normal**: Bell curve
- **Exponential**: Time between events

---

## 🔑 Key Takeaways

| Concept      | Formula                    | Use                     |
| ------------ | -------------------------- | ----------------------- |
| Conditional  | P(A\|B) = P(A∩B)/P(B)      | Events with info        |
| Bayes        | P(A\|B) = P(B\|A)P(A)/P(B) | Inverse probability     |
| Independence | P(A∩B) = P(A)P(B)          | Simplifies calculations |
| Total Prob   | P(A) = ΣP(A\|Bᵢ)P(Bᵢ)      | Marginalizing           |

---

## 📖 Further Reading

- Introduction to Probability by Blitzstein and Hwang
- Pattern Recognition and ML by Bishop (Chapter 1)
