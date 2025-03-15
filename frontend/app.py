import streamlit as st
import json
import subprocess
from models.embedding_model import find_relevant_profiles
from backend.feedback import save_feedback

# Streamlit App Title
st.title("🔍 AI-Powered Profile Search")

# Button to update the FAISS index
if st.button("🔄 Update Data & Rebuild Index"):
    with st.spinner("Updating data..."):
        result = subprocess.run(["python", "backend/process_data.py"], capture_output=True, text=True)
        st.success("✅ Data updated successfully!")
        # st.text_area("Process Output", result.stdout, height=150)

# User Input for Query
query = st.text_input("What are you looking for?", "")

# Custom CSS for tag styling
st.markdown("""
    <style>
        .tag-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 5px;
        }
        .tag {
            display: inline-block;
            padding: 6px 12px;
            font-size: 14px;
            font-weight: bold;
            color: white;
            background-color: #0073e6; /* Default Streamlit blue */
            border-radius: 15px;
            text-align: center;
        }
        @media (prefers-color-scheme: dark) {
            .tag {
                background-color: #ff9800; /* Bright orange in dark mode */
            }
        }
        .spacer {
            margin-bottom: 10px; /* Space between elements */
        }
    </style>
""", unsafe_allow_html=True)

# Processing the Search Query
if query:
    st.write(f"**Searching for:** {query}...")
    results = find_relevant_profiles(query, top_n=5)

    if results:
        st.subheader("📌 Recommended Profiles")

        # Display recommended profiles
        for profile in results:
            with st.expander(f"🔹 {profile.get('name', 'Unknown')} ({profile.get('company', 'Unknown')})"):
                # Longer explanation for why the profile was recommended
                explanation = profile.get('Explanation', 'No explanation available.')
                st.markdown(f"**Why this person?**\n\n{explanation}")

                # Display all generated tags in a button-like style
                tags = profile.get('cleaned_tags', [])
                if tags:
                    st.markdown("**🔖 Tags:**")
                    tag_html = '<div class="tag-container">' + ''.join(
                        f'<span class="tag">{tag}</span>' for tag in tags
                    ) + '</div>'
                    st.markdown(tag_html, unsafe_allow_html=True)
                    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)  # Add spacing

                # Display LinkedIn profile link with additional spacing
                linkedin_url = profile.get('linkedin_url', '').strip()
                if linkedin_url:
                    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)  # Add spacing before link
                    st.markdown(f"**🔗 [View LinkedIn Profile]({linkedin_url})**")

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
