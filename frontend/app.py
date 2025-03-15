import streamlit as st
import json
from models.embedding_model import find_relevant_profiles
from backend.feedback import save_feedback

st.title("🔍 AI-Powered LinkedIn Search")

query = st.text_input("What are you looking for?", "")

if query:
    st.write(f"**Searching for:** {query}...")
    results = find_relevant_profiles(query, top_n=5)

    if results:
        st.subheader("📌 Recommended Profiles")

        # Display recommended profiles with explanations
        for profile in results:
            with st.expander(f"🔹 {profile.get('name', 'Unknown')} ({profile.get('company', 'Unknown')})"):
                st.write(f"**Why this person?** {profile['Explanation']}")
                linkedin_url = profile.get('linkedin_url', '').strip()
                if linkedin_url:  # Only display link if it's available
                    st.write(f"**LinkedIn Profile:** [View Profile]({linkedin_url})")


        # Selection for feedback
        st.subheader("✅ Select the Best Matches")
        selected_profiles = st.multiselect(
            "Which profiles best match what you were looking for?",
            options=[f"{p.get('name', 'Unknown')} - {p.get('company', 'Unknown')}" for p in results] + ["None of these"],
        )

        if st.button("Submit Feedback"):
            selected_data = [
                p for p in results if f"{p.get('name', 'Unknown')} - {p.get('company', 'Unknown')}" in selected_profiles
            ]
            save_feedback(query, selected_data)
            st.success("✅ Feedback submitted! Future searches for similar queries will be improved.")
    else:
        st.write("No relevant profiles found. Try refining your search.")
