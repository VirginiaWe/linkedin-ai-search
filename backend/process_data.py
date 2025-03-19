import re
import pandas as pd
import faiss
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME  # Import model name
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Get absolute path of the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define absolute paths
CSV_FILE = os.path.join(BASE_DIR, "data/profiles.csv")
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

# Function to clean tags
def clean_tags(tag_list):
    SPELLING_CORRECTIONS = {
        "publicspeaker": "Public Speaker",
        "tecnicaloperations": "Technical Operations",
        "machinelearnimg": "Machine Learning",
        "estrategy": "Strategy",
        "advicate": "Advocate",
        "climatefresk": "Climate Fresk",
        "cofunder": "Co-founder",
        "startmaterunclub": "Startmate Run Club",
        "unsw": "UNSW",
        "delloitte": "Deloitte",
        "sw": "Software",
        "energydescentralisation": "Energy Decentralization",
        "impacx": "Impact",
        "mentro": "Mentor",
        "prowomen": "Pro Women",
        "open to work": "Open to Work",
        "reducing scope3": "Reducing Scope 3",
        "startmateclimatetech": "Startmate Climate Tech",
        "excto": "Ex-CTO",
        "exfounder": "Ex-Founder",
        "ceo": "CEO",
        "tcfd": "TCFD",
        "cop31": "COP31",
        "mba": "MBA",
        "msc": "MSc",
        "esg": "ESG",
        "ai": "AI"
    }
    GENERIC_TAGS = {"attendingsxsw2024", "", "and"}

    def clean_tag(tag):
        if not isinstance(tag, str) or tag.strip() == "":
            return None
        tag = re.sub(r"[^\w\s]", "", tag).strip().title()
        tag = SPELLING_CORRECTIONS.get(tag.lower(), tag)
        return tag if tag.lower() not in GENERIC_TAGS else None

    return list(set(filter(None, [clean_tag(tag) for tag in tag_list])))

# Function to generate tags
def generate_tags():
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(df["headline"] + " " + df["about"])
    df["existing_tags"] = df["tags"].apply(lambda x: x.split(",") if x else [])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["headline"] + " " + df["about"], convert_to_numpy=True)
    similarity_matrix = np.dot(embeddings, embeddings.T)

    def get_new_tags(index):
        similar_profiles = np.argsort(similarity_matrix[index])[-5:]
        new_tags = set()
        for similar_idx in similar_profiles:
            new_tags.update(df.iloc[similar_idx]["existing_tags"])
        return list(new_tags)[:7]

    df["generated_tags"] = df.index.to_series().apply(get_new_tags)
    df["final_tags"] = df.apply(lambda row: list(set(row["existing_tags"] + row["generated_tags"])), axis=1)
    df["cleaned_tags"] = df["final_tags"].apply(clean_tags)

# Generate and save tags
generate_tags()

# Save profile metadata in JSON format (including LinkedIn URL)
if not df.empty:
    df.to_json(JSON_FILE, orient="records", indent=4)  # Ensure it's properly formatted
    print("✅ CSV-based embeddings created and JSON metadata saved!")
else:
    print("⚠️ Warning: CSV file is empty. No JSON file saved.")
