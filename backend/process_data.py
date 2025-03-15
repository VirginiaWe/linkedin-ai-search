import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME  # Import model name
from sentence_transformers import SentenceTransformer

# Get absolute path of the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define absolute paths
CSV_FILE = os.path.join(BASE_DIR, "data/linkedin_profiles.csv")
JSON_FILE = os.path.join(BASE_DIR, "data/profiles_metadata.json")
FAISS_FILE = os.path.join(BASE_DIR, "data/embeddings.pkl")

# Check if CSV file exists
if not os.path.exists(CSV_FILE):
    print(f"❌ Error: CSV file '{CSV_FILE}' not found.")
    exit()

df = pd.read_csv(CSV_FILE, dtype=str).fillna("")  # Ensure all data is string and handle NaNs

# Ensure LinkedIn link is included in metadata
if "link" in df.columns:  # Check if 'link' column exists in CSV
    df["linkedin_url"] = df["link"]
else:
    df["linkedin_url"] = ""  # If no 'link' column, set empty string

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Extract descriptions for embeddings
texts = df.apply(lambda row: f"{row['headline']} {row['about']} {row['field']} {row['company']} {row['position']} {row['gender']} {row['tags']}", axis=1).tolist()

# Generate embeddings
embeddings = model.encode(texts, convert_to_numpy=True)

# Store embeddings using FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

print(f"📦 Adding {len(embeddings)} embeddings to FAISS index...")
index.add(embeddings)
print(f"✅ FAISS now contains {index.ntotal} entries.")

# Save FAISS index
faiss.write_index(index, FAISS_FILE)

print("🔍 Sample profile before saving to JSON:", df.iloc[0].to_dict())


# Save profile metadata in JSON format (including LinkedIn URL)
if not df.empty:
    df.to_json(JSON_FILE, orient="records", indent=4)  # Ensure it's properly formatted
    print("✅ CSV-based embeddings created and JSON metadata saved!")
else:
    print("⚠️ Warning: CSV file is empty. No JSON file saved.")
