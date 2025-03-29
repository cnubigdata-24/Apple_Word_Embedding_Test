#pip install sentence-transformers

# Import required libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a pre-trained BERT model (Sentence-BERT)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define two sentences for comparison
sentence_1 = "I ate an apple."
sentence_2 = "Apple Inc. is a technology company."

# Generate 768-dimensional sentence embeddings
embedding_1 = model.encode(sentence_1)
embedding_2 = model.encode(sentence_2)

# Compute cosine similarity between the two embeddings
cosine_sim = cosine_similarity([embedding_1], [embedding_2])[0][0]

# Print results
print("Sentence 1:", sentence_1)
print("Sentence 2:", sentence_2)

# Print first 10 values
print("\nEmbedding 1 (first 10 dimensions):", embedding_1[:10])  
print("Embedding 2 (first 10 dimensions):", embedding_2[:10]) 
print("\nCosine Similarity:", cosine_sim)
