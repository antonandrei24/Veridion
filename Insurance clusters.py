# pip install hdbscan umap-learn scikit-learn sentence-transformers

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
import umap

# 1. Load data
insurance_df = pd.read_csv("insurance.csv", dtype=str).dropna(subset=["label"])
insurance_texts = insurance_df["label"].str.lower().tolist()

# 2. Embed using the same sentence model
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "sentence-transformers/all-mpnet-base-v2"  # or multilingual
model = SentenceTransformer(model_name)

print("Encoding insurance labels...")
embeddings = model.encode(insurance_texts, show_progress_bar=True, convert_to_numpy=True)

# 3. Dimensionality reduction (optional, but helps clustering)
print("Reducing dimensions with UMAP...")
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=10, random_state=42)
embeddings_2d = umap_reducer.fit_transform(embeddings)

# 4a. Option A — KMeans (simple baseline)
# n_clusters = 10  # adjust; start with 10–20
# kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
# labels_km = kmeans.fit_predict(embeddings_2d)
# insurance_df["cluster_kmeans"] = labels_km

# 4b. Option B — HDBSCAN (auto-detect cluster count)
hdb = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, metric='euclidean')
labels_hdb = hdb.fit_predict(embeddings_2d)
insurance_df["cluster_hdbscan"] = labels_hdb

# 5. Inspect clusters
print("\nTop insurance labels by cluster (KMeans):\n")
for cid in sorted(insurance_df["cluster_hdbscan"].unique()):
    examples = insurance_df[insurance_df["cluster_hdbscan"] == cid]["label"].head(8).tolist()
    print(f"Cluster {cid} ({len(examples)} examples):")
    for ex in examples:
        print(f"   • {ex}")
    print()

# 6. Save clustered taxonomy
insurance_df.to_csv("insurance_clustered.csv", index=False)
print("\n✅ Saved insurance_clustered.csv with cluster assignments.")

# --- Load clustered insurance file ---
df = pd.read_csv("insurance_clustered.csv", dtype=str)
df = df.dropna(subset=["label", "cluster_hdbscan"])
df["cluster_hdbscan"] = df["cluster_hdbscan"].astype(int)

# --- Encode labels using the same model ---
model_name = "sentence-transformers/all-mpnet-base-v2"  # or same as before
model = SentenceTransformer(model_name)

print("Encoding insurance labels...")
embeddings = model.encode(df["label"].tolist(), show_progress_bar=True, convert_to_numpy=True)

# --- Compute cluster centroids ---
cluster_ids = sorted(df["cluster_hdbscan"].unique())
centroids = []
for cid in cluster_ids:
    cluster_vectors = embeddings[df["cluster_hdbscan"] == cid]
    centroid = np.mean(cluster_vectors, axis=0)
    centroids.append(centroid)
centroids = np.vstack(centroids)

# --- Compute pairwise cosine distances between centroids ---
dist_matrix = cosine_distances(centroids)

# --- Create a DataFrame for labeling ---
dist_df = pd.DataFrame(dist_matrix, index=[f"Cluster {c}" for c in cluster_ids],
                       columns=[f"Cluster {c}" for c in cluster_ids])

# --- Plot heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(dist_df, annot=True, fmt=".2f", cmap="coolwarm", square=True,
            cbar_kws={"label": "Cosine Distance"})
plt.title("Cluster-to-Cluster Semantic Distances (HDBSCAN)")
plt.tight_layout()
plt.show()