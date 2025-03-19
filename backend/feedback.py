import json
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME


FEEDBACK_FILE = "data/user_feedback.json"

# Load model to embed queries for similarity check
model = SentenceTransformer(MODEL_NAME)

def load_feedback():
    """Load existing feedback from file."""
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"feedback": []}  # Return empty structure if file not found or corrupted

def save_feedback(query, selected_profiles):
    """Save feedback by associating it with similar search queries."""
    feedback_data = load_feedback()

    # Convert query to embedding for similarity tracking
    query_embedding = model.encode([query]).tolist()[0]

    feedback_entry = {
        "query": query,
        "query_embedding": query_embedding,
        "selected_profiles": selected_profiles
    }

    # Append feedback
    feedback_data["feedback"].append(feedback_entry)

    # Save updated feedback
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as file:
        json.dump(feedback_data, file, indent=4)

    print("✅ Feedback saved successfully!")
