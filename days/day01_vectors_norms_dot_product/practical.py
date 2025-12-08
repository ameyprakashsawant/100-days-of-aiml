# Day 1: Vectors, Norms, Dot Product & Projections
# Learning basic vector operations
# Amey Prakash Sawant

import numpy as np
import matplotlib.pyplot as plt
import math

# Working with vectors
print("Day 1: Working with Vectors")
print("="*30)

# Create some vectors
vector1 = [3, 4]
vector2 = [1, 2] 
vector3 = [1, 2, 3, 4, 5]  # 5D vector

print(f"My first vector: {vector1}")
print(f"Second vector: {vector2}")
print(f"Higher dimensional vector: {vector3}")

# Basic vector math
# Adding vectors
result = []
for i in range(len(vector1)):
    result.append(vector1[i] + vector2[i])
print(f"Adding vectors: {vector1} + {vector2} = {result}")

# Multiply by a number
multiplied = []
scale = 3
for num in vector1:
    multiplied.append(num * scale)
print(f"Multiply {vector1} by {scale}: {multiplied}")

# Element by element multiplication
element_mult = []
for i in range(len(vector1)):
    element_mult.append(vector1[i] * vector2[i])
print(f"Element multiplication: {element_mult}")

# Vector norms (lengths)
print("\nVector Lengths (Norms):")
print("-" * 20)

test_vector = [3, -4, 5]

# L1 norm - add up absolute values
l1_length = 0
for num in test_vector:
    l1_length += abs(num)
print(f"L1 length of {test_vector}: {l1_length}")

# L2 norm - square root of sum of squares
l2_length = 0
for num in test_vector:
    l2_length += num * num
l2_length = math.sqrt(l2_length)
print(f"L2 length of {test_vector}: {l2_length:.3f}")

# Max norm - just the biggest absolute value
max_length = 0
for num in test_vector:
    if abs(num) > max_length:
        max_length = abs(num)
print(f"Max length of {test_vector}: {max_length}")

# Make unit vector (length = 1)
unit_vector = []
for num in test_vector:
    unit_vector.append(num / l2_length)
print(f"Unit vector: {[round(x, 3) for x in unit_vector]}")

# Dot product
print("\nDot Product:")
print("-" * 12)

u = [1, 2, 3]
v = [4, 5, 6]

# Calculate dot product manually
dot_product = 0
for i in range(len(u)):
    dot_product += u[i] * v[i]

print(f"u = {u}")
print(f"v = {v}")
print(f"Dot product: {dot_product}")

# Find angle between vectors
def find_angle(vec1, vec2):
    # Calculate dot product
    dot = 0
    for i in range(len(vec1)):
        dot += vec1[i] * vec2[i]
    
    # Calculate lengths
    len1 = math.sqrt(sum(x*x for x in vec1))
    len2 = math.sqrt(sum(x*x for x in vec2))
    
    # cos(angle) = dot / (len1 * len2)
    cos_angle = dot / (len1 * len2)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
angle = find_angle(u, v)
print(f"Angle between vectors: {angle:.1f} degrees")

# Check if vectors are perpendicular (orthogonal)
perpendicular1 = [1, 0]
perpendicular2 = [0, 1]
dot_perp = sum(perpendicular1[i] * perpendicular2[i] for i in range(len(perpendicular1)))
print(f"Perpendicular vectors {perpendicular1} and {perpendicular2}")
print(f"Their dot product: {dot_perp} (should be 0)")

# Vector projections
print("\nProjections:")
print("-" * 11)

a = [3, 4]
b = [5, 0]

# Project vector a onto vector b
# Formula: proj = (a·b / b·b) * b
dot_ab = sum(a[i] * b[i] for i in range(len(a)))
dot_bb = sum(b[i] * b[i] for i in range(len(b)))
projection = [(dot_ab / dot_bb) * b[i] for i in range(len(b))]

print(f"Project {a} onto {b}")
print(f"Projection: {projection}")

# The leftover part should be perpendicular to b
leftover = [a[i] - projection[i] for i in range(len(a))]
dot_leftover_b = sum(leftover[i] * b[i] for i in range(len(leftover)))
print(f"Leftover part: {leftover}")
print(f"Leftover · b = {dot_leftover_b:.6f} (should be ~0)")

# Cosine similarity (used in ML)
print("\nCosine Similarity:")
print("-" * 17)

def cosine_sim(vec1, vec2):
    dot = sum(vec1[i] * vec2[i] for i in range(len(vec1)))
    len1 = math.sqrt(sum(x*x for x in vec1))
    len2 = math.sqrt(sum(x*x for x in vec2))
    return dot / (len1 * len2)

# Example: document similarity
doc1 = [3, 2, 0, 5, 0, 0, 1]  # word counts in document 1
doc2 = [3, 1, 0, 4, 0, 0, 2]  # word counts in document 2 
doc3 = [0, 0, 5, 0, 3, 4, 0]  # word counts in document 3

sim12 = cosine_sim(doc1, doc2)
sim13 = cosine_sim(doc1, doc3)
sim23 = cosine_sim(doc2, doc3)

print("Document similarity:")
print(f"Doc1 vs Doc2: {sim12:.3f}")
print(f"Doc1 vs Doc3: {sim13:.3f}") 
print(f"Doc2 vs Doc3: {sim23:.3f}")

# Simple word similarity example
print("\nWord similarity (using fake word vectors):")
words = {
    'king': [0.8, 0.2, 0.9, 0.1],
    'queen': [0.7, 0.8, 0.9, 0.1], 
    'man': [0.9, 0.1, 0.1, 0.2],
    'woman': [0.8, 0.9, 0.1, 0.2],
    'apple': [0.1, 0.1, 0.2, 0.9],
    'orange': [0.15, 0.1, 0.25, 0.85]
}

def find_similar(word, word_dict):
    target = word_dict[word]
    similarities = []
    
    for other_word, other_vec in word_dict.items():
        if other_word != word:
            sim = cosine_sim(target, other_vec)
            similarities.append((other_word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:2]

print("Most similar words:")
for word in ['king', 'apple']:
    similar = find_similar(word, words)
    print(f"{word}: {similar[0][0]} ({similar[0][1]:.3f}), {similar[1][0]} ({similar[1][1]:.3f})")

# Famous word analogy: king - man + woman ≈ queen
king_vec = words['king']
man_vec = words['man'] 
woman_vec = words['woman']

result_vec = []
for i in range(len(king_vec)):
    result_vec.append(king_vec[i] - man_vec[i] + woman_vec[i])

print(f"\nWord analogy: king - man + woman = ?")
best_match = ""
best_score = -1
for word, vec in words.items():
    if word not in ['king', 'man', 'woman']:
        score = cosine_sim(result_vec, vec)
        if score > best_score:
            best_score = score
            best_match = word

print(f"Answer: {best_match} (score: {best_score:.3f})")

print("\nDay 1 complete! ✅")


print("\n" + "=" * 50)
print("Day 1 Complete! 🎉")
print("=" * 50)
