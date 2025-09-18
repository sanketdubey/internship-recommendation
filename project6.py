import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# optional advanced packages
try:
    from sentence_transformers import SentenceTransformer, util
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    use_semantic = True
except:
    use_semantic = False

try:
    from googletrans import Translator
    translator = Translator()
    use_translate = True
except:
    use_translate = False

# ------------- UI TEXTS IN DIFFERENT LANGUAGES -------------
labels = {
    "English": {
        "title": "AI-Based Internship Recommendation Engine",
        "desc": "Get your top internship recommendations based on your profile.",
        "name": "Your Name",
        "education": "Education / Degree",
        "skills": "Your Skills (comma separated)",
        "sector": "Preferred Sector",
        "location": "Preferred Location",
        "submit": "Recommend Internships",
        "result": "Top Internship Recommendations for"
    },
    "Hindi": {
        "title": "एआई-आधारित इंटर्नशिप अनुशंसा इंजन",
        "desc": "अपनी प्रोफ़ाइल के आधार पर शीर्ष इंटर्नशिप अनुशंसाएं प्राप्त करें।",
        "name": "आपका नाम",
        "education": "शिक्षा / डिग्री",
        "skills": "आपके कौशल (कॉमा से अलग)",
        "sector": "पसंदीदा क्षेत्र",
        "location": "पसंदीदा स्थान",
        "submit": "इंटर्नशिप सुझाएँ",
        "result": "के लिए शीर्ष इंटर्नशिप अनुशंसाएं"
    },
    "Marathi": {
        "title": "एआय-आधारित इंटर्नशिप शिफारस इंजिन",
        "desc": "तुमच्या प्रोफाइलवर आधारित शीर्ष इंटर्नशिप शिफारसी मिळवा.",
        "name": "तुमचे नाव",
        "education": "शिक्षण / पदवी",
        "skills": "तुमचे कौशल्ये (स्वल्पविरामाने वेगळे)",
        "sector": "आवडता विभाग",
        "location": "आवडते ठिकाण",
        "submit": "इंटर्नशिप सुचवा",
        "result": "साठी शीर्ष इंटर्नशिप शिफारसी"
    }
}

# -------------------------- Load internship data --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("internships1.csv")
    df['combined'] = df['RequiredSkills'] + " " + df['Sector'] + " " + df['Location']
    return df

internships = load_data()

# -------------------------- Page Config & CSS --------------------------
st.set_page_config(page_title="InternMatch AI", layout="wide")

# language selector at top
lang = st.selectbox("Choose Language / भाषा निवडा", ["English","Hindi","Marathi"])
text = labels[lang]

st.markdown(f"<h1 style='text-align:center;'>{text['title']}</h1>", unsafe_allow_html=True)
st.write(text['desc'])

st.markdown("""
<style>
.reco-card {
    background-color:#F7F9FC;
    padding:15px;
    border-radius:10px;
    margin-bottom:10px;
    box-shadow:0 4px 8px rgba(0,0,0,0.1);
}
.reco-title {
    color:#007BFF;
    font-size:20px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# -------------------------- Student Form --------------------------
with st.form("student_form"):
    name = st.text_input(text['name'])
    education = st.text_input(text['education'])
    skills = st.text_area(text['skills'])
    sector_pref = st.text_input(text['sector'])
    location_pref = st.text_input(text['location'])
    submitted = st.form_submit_button(text['submit'])

# -------------------------- If form submitted --------------------------
if submitted:
    student_text = skills + " " + sector_pref + " " + location_pref

    # translate input to English for matching
    if use_translate and lang != "en":
        student_text = translator.translate(student_text, src=lang, dest='en').text

    if use_semantic:
        # semantic embeddings
        internship_embeddings = semantic_model.encode(internships['combined'], convert_to_tensor=True)
        student_embedding = semantic_model.encode(student_text, convert_to_tensor=True)
        scores = util.cos_sim(student_embedding, internship_embeddings)[0].cpu().numpy()
    else:
        # TF-IDF fallback
        vectorizer = TfidfVectorizer()
        internship_vectors = vectorizer.fit_transform(internships['combined'])
        student_vector = vectorizer.transform([student_text])
        scores = cosine_similarity(student_vector, internship_vectors)[0]

    internships['score'] = scores
    top3 = internships.sort_values(by='score', ascending=False).head(3)

    st.subheader(f"{text['result']} {name}:")

    for _, row in top3.iterrows():
        st.markdown(
            f"""
            <div class='reco-card'>
                <div class='reco-title'>{row['Title']}</div>
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

    if st.button("Save these Recommendations (future scope)"):
        st.success("Recommendations saved locally (future scope)")

st.info("This is an advanced multilingual prototype. Future scope: REST API with Flask, PostgreSQL DB, React frontend, "
        "regional languages & SendGrid/Twilio notifications.")
