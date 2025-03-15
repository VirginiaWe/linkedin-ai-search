import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME

FEEDBACK_FILE = "data/user_feedback.json"
DATA_FILE = "data/profiles_metadata.json"

# Load AI model
model = SentenceTransformer(MODEL_NAME)

# Check if JSON file exists and is not empty
if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
    try:
        df = pd.read_json(DATA_FILE)
    except ValueError:
        print("⚠️ Error reading JSON file. It may be corrupted. Try regenerating it.")
        df = pd.DataFrame()  # Use an empty DataFrame to prevent crashes
else:
    print("⚠️ JSON file is missing or empty. Please run process_data.py to regenerate it.")
    df = pd.DataFrame()  # Prevents crashes if file is empty

# Extract relevant columns for embeddings (only if DataFrame is not empty)
if not df.empty:
    profile_texts = df.apply(lambda row: f"{row['headline']} {row['about']} {row['field']} {row['company']} {row['position']} {row['gender']} {row['tags']}", axis=1).tolist()
    profile_embeddings = model.encode(profile_texts, convert_to_numpy=True)
else:
    profile_embeddings = np.array([])  # Prevents errors in similarity calculations

def load_feedback():
    """Load existing feedback data."""
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as file:
            return json.load(file).get("feedback", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def generate_explanation(profile, query):
    """Creates a short explanation of why this person was recommended."""
    explanation_parts = []

    if profile.get("position"):
        explanation_parts.append(f"Works as **{profile['position']}**")

    if profile.get("company"):
        explanation_parts.append(f"at **{profile['company']}**")

    if profile.get("about"):
        explanation_parts.append(f"Profile mentions: *{profile['about'][:100]}...*")  # Short preview

    if profile.get("field"):
        explanation_parts.append(f"Field: **{profile['field']}**")

    return " | ".join(explanation_parts) if explanation_parts else "Relevant based on profile analysis"

def find_relevant_profiles(query, top_n=5, base_threshold=0.4):
    """Find relevant profiles dynamically based on similarity scores and feedback."""
    if df.empty or profile_embeddings.size == 0:
        return []  # No profiles to search if data is missing

    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, profile_embeddings)[0]

    # Sort all matches by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    matched_profiles = df.iloc[sorted_indices].to_dict(orient="records")

    # **Dynamically adjust the similarity threshold**
    max_similarity = similarities[sorted_indices[0]]  # Highest similarity score
    adaptive_threshold = max(base_threshold, max_similarity * 0.6)  # 60% of best match

    print(f"🔍 Adaptive similarity threshold: {adaptive_threshold:.2f}")

    # Filter profiles based on the dynamic threshold
    filtered_profiles = [
        (profile, similarities[i]) for i, profile in zip(sorted_indices, matched_profiles)
        if similarities[i] >= adaptive_threshold
    ]

    # **Ensure we return at least `top_n` results as fallback**
    if len(filtered_profiles) < top_n:
        filtered_profiles = [
            (profile, similarities[i]) for i, profile in zip(sorted_indices[:top_n], matched_profiles[:top_n])
        ]

    # Load feedback to adjust ranking
    feedback_data = load_feedback()
    feedback_boost = {}

    # Find feedback for **similar queries**
    for feedback_entry in feedback_data:
        past_query_embedding = np.array(feedback_entry["query_embedding"]).reshape(1, -1)
        similarity_score = cosine_similarity(query_embedding.reshape(1, -1), past_query_embedding)[0][0]

        if similarity_score > 0.85:  # Use feedback only if query similarity is high
            for profile in feedback_entry["selected_profiles"]:
                profile_name = profile["name"]
                feedback_boost[profile_name] = feedback_boost.get(profile_name, 0) + 1  # Increment boost

    # Re-rank matched profiles based on feedback
    for profile, score in filtered_profiles:
        profile["boost_score"] = feedback_boost.get(profile["name"], 0)  # Apply boost

    # Sort profiles by feedback boost + similarity score
    filtered_profiles.sort(key=lambda p: (p[0]["boost_score"], p[1]), reverse=True)

    # ✅ Add explanations before returning results
    results = []
    for profile, _ in filtered_profiles:
        profile["Explanation"] = generate_explanation(profile, query)
        results.append(profile)

    return results
