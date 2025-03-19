import faiss
import json
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME

# Load model and FAISS index
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index("data/embeddings.pkl")

# Load metadata
df = pd.read_json("data/profiles_metadata.json")  # Now JSON created from CSV

def search_profiles(query, top_k=5):
    """Search LinkedIn profiles based on a natural language query."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    print(f"🔍 Query embedding shape: {query_embedding.shape}")

    distances, indices = index.search(query_embedding, top_k)
    print(f"📊 FAISS returned indices: {indices}")

    results = df.iloc[indices[0]].to_dict(orient="records") if len(indices[0]) > 0 else []

    if results:
        print(f"✅ Found {len(results)} results for '{query}'")
    else:
        print(f"❌ No results found for '{query}'")

    return results
