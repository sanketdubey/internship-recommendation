import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --------------------------
# Load internship data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("internships1.csv")
    # combine text features for similarity
    df['combined'] = df['RequiredSkills'] + " " + df['Sector'] + " " + df['Location']
    return df

internships = load_data()

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="InternMatch AI", layout="wide")
st.markdown("<h1 style='text-align:center;'>AI-Based Internship Recommendation Engine</h1>", unsafe_allow_html=True)
st.write("Get your top internship recommendations based on your profile.")

# --------------------------
# Student Form
# --------------------------
with st.form("student_form"):
    name = st.text_input("Your Name")
    education = st.text_input("Education / Degree")
    skills = st.text_area("Your Skills (comma separated)")
    sector = st.text_input("Preferred Sector")
    location = st.text_input("Preferred Location")
    submitted = st.form_submit_button("Recommend Internships")

# --------------------------
# If form submitted
# --------------------------
if submitted:
    # Combine student input into one string
    student_text = skills + " " + sector + " " + location

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    internship_vectors = vectorizer.fit_transform(internships['combined'])
    student_vector = vectorizer.transform([student_text])

    # Cosine Similarity Scores
    scores = cosine_similarity(student_vector, internship_vectors)[0]
    internships['score'] = scores

    # Top 3 recommendations
    top3 = internships.sort_values(by='score', ascending=False).head(3)

    st.subheader(f"Top Internship Recommendations for {name}:")

    # --------------------------
    # Display cards
    # --------------------------
    for _, row in top3.iterrows():
        st.markdown(
            f"""
            <div style="background-color:#F7F9FC;
                        padding:15px;
                        border-radius:10px;
                        margin-bottom:10px;">
                <h4 style="color:#007BFF;margin-bottom:5px;">{row['Title']}</h4>
                <p style="color:#000000;margin:0;">
                    <b>Skills Required:</b> {row['RequiredSkills']}
                </p>
                <p style="color:#000000;margin:0;">
                    <b>Sector:</b> {row['Sector']} | <b>Location:</b> {row['Location']}
                </p>
                <p style="color:#000000;margin:0;">
                    <b>Match Score:</b> {round(row['score']*100,2)}%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.button("Send Recommendations by Email (future scope)")

# --------------------------
# Info footer
# --------------------------
st.info(
    "This is an advanced prototype. "
    "Future scope: REST API with Flask, PostgreSQL DB, React frontend, "
    "regional languages & SendGrid/Twilio notifications."
)

