import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load internship data
internships = pd.read_csv("internships.csv")
internships['combined'] = internships['RequiredSkills'] + " " + internships['Sector']

st.title("AI-Based Internship Recommendation Engine")
st.write("Enter your details to get top 3 internship matches")

# User input form
with st.form("student_form"):
    name = st.text_input("Your Name")
    education = st.text_input("Education / Degree")
    skills = st.text_area("Your Skills (comma separated)")
    sector = st.text_input("Preferred Sector")
    location = st.text_input("Preferred Location")
    submitted = st.form_submit_button("Recommend Internships")

if submitted:
    # Combine student skills + sector
    student_text = skills + " " + sector

    # Vectorize
    vectorizer = TfidfVectorizer()
    internship_vectors = vectorizer.fit_transform(internships['combined'])
    student_vector = vectorizer.transform([student_text])

    # Compute similarity
    scores = cosine_similarity(student_vector, internship_vectors)[0]
    internships['score'] = scores

    # Top 3 internships
    top3 = internships.sort_values(by='score', ascending=False).head(3)

    st.subheader("Top 3 Internship Recommendations:")
    for idx, row in top3.iterrows():
        st.markdown(f"### {row['Title']}")
        st.write(f"**Skills Required:** {row['RequiredSkills']}")
        st.write(f"**Sector:** {row['Sector']} | **Location:** {row['Location']}")
        st.write(f"**Match Score:** {round(row['score']*100,2)}%")
        st.markdown("---")
