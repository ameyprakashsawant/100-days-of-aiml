# Project 3: Recommendation System
# Making suggestions like Netflix and Spotify do!
# Amey Prakash Sawant

print("Project 3: Recommendation System")
print("=" * 33)

# How similar are two things? Let's find out!

def cosine_similarity(list1, list2):
    # How similar are two lists? (like movie preferences)
    # cosine similarity = dot product / (length1 * length2)
    
    # Step 1: Dot product (multiply matching elements and add up)
    dot_product = 0
    for i in range(len(list1)):
        dot_product += list1[i] * list2[i]
    
    # Step 2: Calculate lengths (magnitudes)
    length1 = 0
    for x in list1:
        length1 += x * x
    length1 = length1 ** 0.5
    
    length2 = 0
    for x in list2:
        length2 += x * x
    length2 = length2 ** 0.5
    
    # Step 3: Avoid division by zero
    if length1 == 0 or length2 == 0:
        return 0.0
    
    # Step 4: Final similarity score
    return dot_product / (length1 * length2)
def euclidean_distance(list1, list2):
    # How far apart are two things? (smaller = more similar)
    distance = 0
    for i in range(len(list1)):
        diff = list1[i] - list2[i]
        distance += diff * diff
    return distance ** 0.5

def euclidean_similarity(list1, list2):
    # Convert distance to similarity (1 = identical, 0 = very different)
    distance = euclidean_distance(list1, list2)
    return 1 / (1 + distance)

# Let's test these similarity functions!
print("\nTesting similarity functions:")
print("-" * 30)

# Two users' movie ratings (1-5 scale)
alice_ratings = [5, 4, 1, 2, 5]  # Likes action, dislikes romance
bob_ratings = [4, 5, 2, 1, 4]    # Similar taste to Alice
charlie_ratings = [1, 2, 5, 4, 1]  # Opposite taste

print("Movie ratings (Action, Comedy, Romance, Drama, SciFi):")
print(f"Alice:   {alice_ratings}")
print(f"Bob:     {bob_ratings}")  
print(f"Charlie: {charlie_ratings}")

print("\nSimilarities:")
alice_bob_sim = cosine_similarity(alice_ratings, bob_ratings)
alice_charlie_sim = cosine_similarity(alice_ratings, charlie_ratings)
print(f"Alice-Bob:     {alice_bob_sim:.3f} (similar taste)")
print(f"Alice-Charlie: {alice_charlie_sim:.3f} (different taste)")

# Content-based recommendation: recommend based on movie features
print("\n" + "=" * 50)
print("CONTENT-BASED RECOMMENDATIONS")
print("=" * 50)

# Movies with features [Action, Comedy, Romance, SciFi, Drama] (0-1 scale)
movies = {
    "Avengers": [1.0, 0.2, 0.0, 0.3, 0.1],
    "Inception": [0.7, 0.1, 0.1, 0.8, 0.5], 
    "Notebook": [0.0, 0.1, 1.0, 0.0, 0.8],
    "Deadpool": [0.8, 0.9, 0.1, 0.2, 0.1],
    "Matrix": [1.0, 0.0, 0.1, 1.0, 0.3]
}

def find_similar_movies(target_movie, movies_dict, top_n=3):
    target_features = movies_dict[target_movie]
    similarities = {}
    
    for movie, features in movies_dict.items():
        if movie != target_movie:
            sim = cosine_similarity(target_features, features)
            similarities[movie] = sim
    
    # Sort by similarity (highest first)
    sorted_movies = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_movies[:top_n]

print("Movie features (Action, Comedy, Romance, SciFi, Drama):")
for movie, features in movies.items():
    print(f"{movie:10}: {features}")

target = "Avengers"
similar_movies = find_similar_movies(target, movies)
print(f"\nSimilar to '{target}':")
for movie, similarity in similar_movies:
    print(f"  {movie}: {similarity:.3f}")
# Collaborative filtering: "People like you also liked..."
print("\n" + "=" * 50) 
print("COLLABORATIVE FILTERING")
print("=" * 50)

# User-item rating matrix (0 = not rated)
users = ["Alice", "Bob", "Charlie", "Diana"]
items = ["Movie1", "Movie2", "Movie3", "Movie4", "Movie5"]

# Rating matrix: rows=users, columns=movies
ratings = [
    [5, 4, 0, 0, 1],  # Alice likes Movie1, Movie2, not Movie5
    [4, 5, 2, 0, 1],  # Bob similar to Alice  
    [0, 0, 5, 4, 0],  # Charlie likes different movies
    [1, 0, 4, 5, 0],  # Diana similar to Charlie
]

print("Rating matrix (0 = not watched):")
print("         ", end="")
for item in items:
    print(f"{item:>8}", end="")
print()

for i, user in enumerate(users):
    print(f"{user:>8} ", end="")
    for rating in ratings[i]:
        if rating > 0:
            print(f"{rating:>8}", end="")
        else:
            print("       -", end="")
    print()

def find_user_similarity(user1_idx, user2_idx, rating_matrix):
    # Find movies both users rated
    user1_ratings = rating_matrix[user1_idx]
    user2_ratings = rating_matrix[user2_idx]
    
    # Get ratings for movies both users watched
    common_ratings1 = []
    common_ratings2 = []
    
    for i in range(len(user1_ratings)):
        if user1_ratings[i] > 0 and user2_ratings[i] > 0:
            common_ratings1.append(user1_ratings[i])
            common_ratings2.append(user2_ratings[i])
    
    if len(common_ratings1) == 0:
        return 0.0  # No common movies
    
    return cosine_similarity(common_ratings1, common_ratings2)

print(f"\nUser similarities:")
for i in range(len(users)):
    for j in range(i+1, len(users)):
        sim = find_user_similarity(i, j, ratings)
        print(f"{users[i]}-{users[j]}: {sim:.3f}")

def predict_rating_collaborative(user_idx, item_idx, rating_matrix, user_names):
    # Predict what user will rate an item based on similar users
    target_ratings = rating_matrix[user_idx]
    
    if target_ratings[item_idx] > 0:
        return target_ratings[item_idx]  # Already rated
    
    # Find similar users who rated this item
    numerator = 0
    denominator = 0
    
    for other_user in range(len(rating_matrix)):
        if other_user != user_idx and rating_matrix[other_user][item_idx] > 0:
            similarity = find_user_similarity(user_idx, other_user, rating_matrix)
            if similarity > 0:
                numerator += similarity * rating_matrix[other_user][item_idx]
                denominator += similarity
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

# Make a prediction for Alice on Movie3
alice_idx = 0
movie3_idx = 2
predicted = predict_rating_collaborative(alice_idx, movie3_idx, ratings, users)
print(f"\nPredicted rating for {users[alice_idx]} on {items[movie3_idx]}: {predicted:.2f}")

# Find recommendations for Alice
print(f"\nRecommendations for {users[alice_idx]}:")
for item_idx in range(len(items)):
    if ratings[alice_idx][item_idx] == 0:  # Not yet watched
        pred = predict_rating_collaborative(alice_idx, item_idx, ratings, users)
        if pred > 0:
            print(f"  {items[item_idx]}: {pred:.2f}")

# Matrix factorization (simplified approach)
print("\n" + "=" * 50)
print("MATRIX FACTORIZATION (Simple)")
print("=" * 50)

print("Matrix factorization finds hidden patterns:")
print("- Maybe there's a 'Action Movie Lover' factor")
print("- Or a 'Comedy Fan' factor") 
print("- Each user has some amount of each factor")
print("- Each movie appeals to certain factors")

# Simple example: assume 2 hidden factors
print(f"\nOriginal ratings matrix:")
for i, user in enumerate(users):
    print(f"{user}: {ratings[i]}")

print(f"\nImagine each user has 2 hidden preferences:")
print("Factor 1: Action movies, Factor 2: Romance movies")

# Simplified user factors [Action_preference, Romance_preference]
user_factors = [
    [0.9, 0.1],  # Alice loves action
    [0.8, 0.2],  # Bob likes action too
    [0.2, 0.9],  # Charlie loves romance  
    [0.3, 0.8],  # Diana likes romance
]

# Movie factors [Action_content, Romance_content]
movie_factors = [
    [0.9, 0.1],  # Movie1: action movie
    [0.8, 0.2],  # Movie2: action with some romance
    [0.1, 0.9],  # Movie3: romance movie
    [0.2, 0.8],  # Movie4: mostly romance
    [0.5, 0.5],  # Movie5: balanced
]

print(f"\nUser preferences:")
for i, user in enumerate(users):
    print(f"{user}: Action={user_factors[i][0]:.1f}, Romance={user_factors[i][1]:.1f}")

print(f"\nMovie content:")
for i, movie in enumerate(items):
    print(f"{movie}: Action={movie_factors[i][0]:.1f}, Romance={movie_factors[i][1]:.1f}")

def predict_rating_mf(user_idx, movie_idx, user_factors, movie_factors):
    # Dot product of user preferences and movie content
    rating = 0
    for f in range(len(user_factors[user_idx])):
        rating += user_factors[user_idx][f] * movie_factors[movie_idx][f]
    return rating * 5  # Scale to 1-5

print(f"\nPredicted ratings using matrix factorization:")
print("         ", end="")
for movie in items:
    print(f"{movie:>8}", end="")
print()

for i, user in enumerate(users):
    print(f"{user:>8} ", end="")
    for j in range(len(items)):
        pred = predict_rating_mf(i, j, user_factors, movie_factors)
        print(f"{pred:>8.1f}", end="")
    print()

print("\n" + "=" * 50)
print("SUMMARY - Three Recommendation Approaches:")
print("=" * 50)
print("1. Content-Based: 'Movies similar to what you liked'")
print("   - Uses movie features (genre, actors, etc.)")
print("   - Good for new users, explains recommendations")
print("   - Limited by feature quality")

print("\n2. Collaborative Filtering: 'People like you also liked'")
print("   - Uses only rating patterns")
print("   - Can discover surprising connections") 
print("   - Needs lots of users and ratings")

print("\n3. Matrix Factorization: 'Hidden preference patterns'")
print("   - Finds latent factors automatically")
print("   - Handles sparse data well")
print("   - Less interpretable but often more accurate")

print("\nProject 3 complete! ✅")
print("\nReal systems like Netflix use hybrid approaches")
print("combining all three methods for best results!")
