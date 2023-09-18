import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def nmf(V, k, max_iter=100, tol=1e-10):
    m, n = V.shape
    # Initialize W and H with random nonnegative values
    W = np.abs(np.random.randn(m, k))
    H = np.abs(np.random.randn(k, n))
    for _ in range(max_iter):
        
        H = H * (W.T @ V) / (W.T @ W @ H)
        W = W * (V @ H.T) / (W @ H @ H.T)
        
        # Convergence check
        if np.linalg.norm(V - W @ H) < tol:
            break
    return W, H

def main():

    np.random.seed(0)
    
    # Sample corpus (collection of documents)
    corpus = [
        "Apple releases new iPhone",
        "Samsung announces Galaxy phone",
        "Apple and Samsung compete in the smartphone market",
        "NASA launches new satellite",
        "SpaceX sends astronauts to the space station",
        "Apple and NASA collaborate on space technology"
    ]

    # Convert the corpus into a term-document matrix using TF-IDF weighting
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)

    # Apply NMF
    # Assume there are three topics
    W, H = nmf(X, k=3)

    # Print the most significant terms for each topic
    terms = vectorizer.get_feature_names_out()
    for i, topic in enumerate(H):
        print(f"Topic {i+1}:")
        print(" ".join([terms[j] for j in topic.argsort()[-5:]]))
        print()

    # For each document, print the most significant topic
    for i, doc in enumerate(corpus):
        print(f"Document {i+1} ({doc}) is mostly about Topic {np.argmax(W[i])+1}")
        print()

if __name__ == "__main__":
    main()