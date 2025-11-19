import numpy as np

def cosine_similarity(v1, v2):
    """Calculate the cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1) #calculating the magnitude of vector using l2 norm, or euclidean norm, or length of vector, or square root of sum of squares of each element
    norm2 = np.linalg.norm(v2)
    return dot /(norm1 * norm2)

a = np.array([1, 2, 3]) ## main array on which we will compare other arrays

# Vectors in the same direction
b = np.array([2, 4, 6])
print("Cosine Similarity value for sam direction:", cosine_similarity(a, b))

# Vectors in the opposite direction
c = np.array([-1, -2, -3])
print("Cosine Similarity value for opp direction:", cosine_similarity(a, c))

# Orthogonal vectors or perpendicular vectors
d = np.array([0, 3, -2])
print("Cosine Similarity value for orthogonal direction:", cosine_similarity(a, d))

#housing data example
house1 = np.array([2500, 4, 3, 2]) # size, bedrooms, bathrooms, floors
house2 = np.array([3000, 5, 4, 3]) # similar houses
house3 = np.array([800, 2, 1, 1]) # smaller house or different type

print("Cosine Similarity between house1 and house2:", cosine_similarity(house1, house2))
print("Cosine Similarity between house1 and house3:", cosine_similarity(house1, house3))


# Text data example using word embeddings
word1 = np.array([0.1, 0.3, 0.5]) # embedding for "king"
word2 = np.array([0.1, 0.3, 0.4]) # embedding for "queen"
word3 = np.array([0.4, 0.2, 0.1]) # embedding for "car"
print("Cosine Similarity between 'king' and 'queen':", cosine_similarity(word1, word2))
print("Cosine Similarity between 'king' and 'car':", cosine_similarity(word1, word3))
