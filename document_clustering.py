'''
Document Clustering Project
===========================
This script demonstrates a moderate-complexity pipeline to group text documents into topic-based clusters.
We compare two text vectorization techniques: Count Vectorizer and TF-IDF Vectorizer.
We then apply K-Means clustering and evaluate cluster coherence using silhouette scores and interpret clusters via top terms.
Print statements have been added to track progress, and we use the smaller 'train' subset for quicker runs during testing.
Includes a retry mechanism to handle incomplete 20 Newsgroups downloads.
'''

# 1. Imports
from sklearn.datasets import fetch_20newsgroups, get_data_home
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import shutil, os

# 2. Load Sample Data
print("Loading 20 Newsgroups training data or custom PDFs...")

# Option A: use built-in 20 Newsgroups dataset for quick testing
use_builtin = True  # set to False to load your own PDFs

if use_builtin:
    categories = ['sci.space', 'rec.autos', 'comp.graphics', 'talk.politics.mideast']
    try:
        data = fetch_20newsgroups(subset='train', categories=categories,
                                  remove=('headers', 'footers', 'quotes'))
    except Exception as e:
        print("Error fetching 20 Newsgroups data, clearing cache and retrying...")
        cache_dir = os.path.join(get_data_home(), '20news_home')
        shutil.rmtree(cache_dir, ignore_errors=True)
        data = fetch_20newsgroups(subset='train', categories=categories,
                                  remove=('headers', 'footers', 'quotes'))
    docs = data.data
else:
    # Option B: load all PDF files from a folder
    from PyPDF2 import PdfReader

    pdf_folder = '/Users/samarthgurlhosur/Desktop/DMDW Presentation/venv'  # change this to your folder path
    docs = []
    for fname in os.listdir(pdf_folder):
        if fname.lower().endswith('.pdf'):
            reader = PdfReader(os.path.join(pdf_folder, fname))
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
            docs.append(text)

print(f"Number of documents loaded: {len(docs)}")

# 3. Preprocessing (basic)
print("Preprocessing documents...")
def preprocess(texts):
    cleaned = [doc.lower() for doc in texts if len(doc.split()) > 5]
    return cleaned

docs_clean = preprocess(docs)
print(f"Documents after cleaning: {len(docs_clean)}")

# 4. Vectorization Techniques
def vectorize(texts, method='count'):
    print(f"Vectorizing text using {method.upper()}...")
    if method == 'count':
        vec = CountVectorizer(max_df=0.8, min_df=5, stop_words='english')
    elif method == 'tfidf':
        vec = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
    else:
        raise ValueError("Method must be 'count' or 'tfidf'.")
    matrix = vec.fit_transform(texts)
    print(f"Vectorization complete: {matrix.shape[0]} docs, {matrix.shape[1]} features")
    return matrix, vec

# 5. Clustering & Evaluation
def cluster_and_evaluate(matrix, n_clusters=4, random_state=42):
    print("Clustering with K-Means...")
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(matrix)
    sil_score = silhouette_score(matrix, labels)
    print(f"Silhouette Score: {sil_score:.4f}")
    return model, labels, sil_score

# 6. Display Top Terms per Cluster
def print_top_terms(model, vectorizer, top_n=10):
    terms = vectorizer.get_feature_names_out()
    centroids = model.cluster_centers_
    for idx, centroid in enumerate(centroids):
        top_indices = centroid.argsort()[-top_n:][::-1]
        top_terms = [terms[i] for i in top_indices]
        print(f"Cluster {idx}: {', '.join(top_terms)}")

# 7. Main Execution
if __name__ == '__main__':
    print("\nRunning clustering pipeline...")
    results = []
    for method in ['count', 'tfidf']:
        matrix, vec = vectorize(docs_clean, method)
        model, labels, sil = cluster_and_evaluate(matrix)
        print_top_terms(model, vec)
        results.append({'method': method, 'silhouette': sil})
        print("-----------------------------------")

    summary = pd.DataFrame(results)
    print("\nSummary of Results:")
    print(summary)
