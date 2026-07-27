import nltk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Load SBERT model (pretrained transformer for sentence embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Input headlines
headlines = []
n = int(input("Enter number of headlines: "))
for i in range(n):
    headlines.append(input(f"Enter headline {i+1}: "))

# Convert headlines to embeddings
X = model.encode(headlines)

# Cosine similarity matrix
print("\nCosine Similarity Matrix:")
print(cosine_similarity(X))

# DBSCAN clustering
# Important: DBSCAN with metric='cosine' requires precomputed distance matrix OR metric='precomputed'.
# Instead, use metric='cosine' directly with embeddings.
dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
labels = dbscan.fit_predict(X)

print("\nHeadline Clusters:")
for i, headline in enumerate(headlines):
    print(f"{headline} -> Cluster {labels[i]}")

# Word similarity using SBERT embeddings
w1 = input("\nEnter first word: ")
w2 = input("Enter second word: ")

# Encode words
vec1 = model.encode([w1])[0]
vec2 = model.encode([w2])[0]

# Cosine similarity between word embeddings
sim = cosine_similarity([vec1], [vec2])[0][0]
print("Embedding-based Word Similarity:", sim)
