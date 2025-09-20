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
        "desc": "Get your top internship recommendations based on your profile.",
        "name": "Your Name (e.g. John Doe)",
        "education": "Education / Degree (e.g. B.Tech CSE 3rd year)",
        "skills": "Your Skills (comma separated) (e.g. Python, SQL, Data Analysis)",
        "sector": "Preferred Sector (e.g. Data Science, Marketing)",
        "location": "Preferred Location (e.g. Mumbai, Remote)",
        "submit": "Recommend Internships",
        "result": "Top Internship Recommendations for"
    },
    "Hindi": {
        "desc": "अपनी प्रोफ़ाइल के आधार पर शीर्ष इंटर्नशिप अनुशंसाएं प्राप्त करें।",
        "name": "आपका नाम (उदा. राहुल शर्मा)",
        "education": "शिक्षा / डिग्री (उदा. बी.टेक सीएसई तीसरा वर्ष)",
        "skills": "आपके कौशल (कॉमा से अलग) (उदा. पायथन, एसक्यूएल, डेटा विश्लेषण)",
        "sector": "पसंदीदा क्षेत्र (उदा. डेटा साइंस, मार्केटिंग)",
        "location": "पसंदीदा स्थान (उदा. मुंबई, रिमोट)",
        "submit": "इंटर्नशिप सुझाएँ",
        "result": "के लिए शीर्ष इंटर्नशिप अनुशंसाएं"
    },
    "Marathi": {
        "desc": "तुमच्या प्रोफाइलवर आधारित शीर्ष इंटर्नशिप शिफारसी मिळवा.",
        "name": "तुमचे नाव (उदा. राहुल पाटील)",
        "education": "शिक्षण / पदवी (उदा. बी.टेक CSE तिसरे वर्ष)",
        "skills": "तुमचे कौशल्ये (स्वल्पविरामाने वेगळे) (उदा. Python, SQL, Data Analysis)",
        "sector": "आवडता विभाग (उदा. Data Science, Marketing)",
        "location": "आवडते ठिकाण (उदा. मुंबई, रिमोट)",
        "submit": "इंटर्नशिप सुचवा",
        "result": "साठी शीर्ष इंटर्नशिप शिफारसी"
    }
}

# multilingual FAQ text
faq = {
    "English":[
        ("What is InternMate?","InternMate is an AI-based engine that recommends internships to students based on their skills, sector and location preferences."),
        ("How does the recommendation work?","We analyze your entered skills, sector, and location, then match them with internships using AI (semantic embeddings/TF-IDF)."),
        ("Do I need to pay to use InternMate?","No. This is a free prototype for students."),
        ("How many internships will I see?","Currently, we show your top 3 matching internships, but this can be increased later."),
        ("Can I get notifications on Email or SMS?","Yes. Email notifications are available; SMS will require phone number verification on Twilio."),
        ("How to change the language?","Use the dropdown at the top to select English, Hindi or Marathi.")
    ],
    "Hindi":[
        ("InternMate क्या है?","InternMate एक AI आधारित इंजन है जो छात्रों को उनके कौशल, क्षेत्र और स्थान के आधार पर इंटर्नशिप सुझाता है।"),
        ("सिफ़ारिश कैसे काम करती है?","हम आपके कौशल, क्षेत्र और स्थान को देखते हैं और AI से मैच करते हैं।"),
        ("क्या InternMate का उपयोग करने के लिए भुगतान करना होगा?","नहीं। यह छात्रों के लिए एक मुफ्त प्रोटोटाइप है।"),
        ("मुझे कितनी इंटर्नशिप दिखाई देंगी?","अभी हम शीर्ष 3 इंटर्नशिप दिखाते हैं, बाद में इसे बढ़ाया जा सकता है।"),
        ("क्या मुझे ईमेल या एसएमएस नोटिफिकेशन मिल सकते हैं?","हाँ, ईमेल नोटिफिकेशन उपलब्ध हैं; एसएमएस के लिए Twilio पर नंबर वेरिफिकेशन चाहिए।"),
        ("भाषा कैसे बदलें?","ऊपर ड्रॉपडाउन से भाषा चुनें।")
    ],
    "Marathi":[
        ("InternMate काय आहे?","InternMate हे AI आधारित इंजिन आहे जे विद्यार्थ्यांना त्यांच्या कौशल्य, विभाग आणि ठिकाणावरून इंटर्नशिप सुचवते."),
        ("शिफारस कशी कार्य करते?","आम्ही तुमचे कौशल्य, विभाग आणि ठिकाण पाहतो व AI ने जुळवतो."),
        ("InternMate वापरण्यासाठी पैसे द्यावे लागतील का?","नाही. हे विद्यार्थ्यांसाठी मोफत प्रोटोटाइप आहे."),
        ("मला किती इंटर्नशिप दिसतील?","सध्या आम्ही तुमच्या टॉप 3 इंटर्नशिप दाखवतो, नंतर वाढवू शकतो."),
        ("मला ईमेल किंवा एसएमएस नोटिफिकेशन मिळू शकते का?","होय, ईमेल नोटिफिकेशन उपलब्ध आहेत; एसएमएससाठी Twilio वर नंबर व्हेरिफिकेशन लागेल."),
        ("भाषा कशी बदलायची?","वरच्या ड्रॉपडाउनमधून भाषा निवडा.")
    ]
}

# -------------------------- Load internship data --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("internships1.csv")
    df['combined'] = df['RequiredSkills'] + " " + df['Sector'] + " " + df['Location']
    return df

internships = load_data()

# -------------------------- Page Config & CSS --------------------------
st.set_page_config(page_title="InternMate – AI Internship Recommendation", layout="wide")

# ✅ Heading above language selector
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #00C3FF;
        font-size: 45px;
        font-weight: 800;
        margin-bottom: 0px;
        font-family: inherit;
    }
    .subtitle {
        text-align: center;
        color: #cccccc;
        font-size: 18px;
        margin-top: 5px;
        margin-bottom: 30px;
        font-family: inherit;
    }
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

st.markdown("<div class='main-title'>InternMate</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Based Internship Recommendation Engine</div>", unsafe_allow_html=True)

# language selector below heading
lang = st.selectbox("Choose Language / भाषा निवडा", ["English","Hindi","Marathi"])
text = labels[lang]

# apply regional font automatically
if lang == "English":
    st.markdown("<style>body{font-family:'Roboto','Segoe UI',sans-serif;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{font-family:'Noto Sans Devanagari','Mangal',sans-serif;}</style>", unsafe_allow_html=True)

st.write(text['desc'])

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
    if use_translate and lang != "English":
        student_text = translator.translate(student_text, src=lang, dest='en').text

    with st.spinner("🔎 Searching for best internships..."):
        if use_semantic:
            internship_embeddings = semantic_model.encode(internships['combined'], convert_to_tensor=True)
            student_embedding = semantic_model.encode(student_text, convert_to_tensor=True)
            scores = util.cos_sim(student_embedding, internship_embeddings)[0].cpu().numpy()
        else:
            vectorizer = TfidfVectorizer()
            internship_vectors = vectorizer.fit_transform(internships['combined'])
            student_vector = vectorizer.transform([student_text])
            scores = cosine_similarity(student_vector, internship_vectors)[0]

        internships['score'] = scores
        top3 = internships.sort_values(by='score', ascending=False).head(3)

    st.success("✅ Recommendations ready!")
    st.subheader(f"{text['result']} {name}:")

    for _, row in top3.iterrows():
        st.markdown(
            f"""
            <div class='reco-card'>
                <div class='reco-title'>{row['Title']}</div>
                <p style="color:#000000;margin:0;">
                    <b>Skills Required:</b> {row['RequiredSkills']}</p>
                <p style="color:#000000;margin:0;">
                    <b>Sector:</b> {row['Sector']} | <b>Location:</b> {row['Location']}</p>
                <p style="color:#000000;margin:0;">
                    <b>Match Score:</b> {round(row['score']*100,2)}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.button("Save these Recommendations (future scope)"):
        st.success("Recommendations saved locally (future scope)")

# -------------------- Help / FAQ Section --------------------
st.markdown("## ❓ Help / FAQ")
for i,(q,a) in enumerate(faq[lang],start=1):
    with st.expander(f"{i}️⃣ {q}"):
        st.write(a)

st.info("This is an advanced multilingual prototype. Future scope: REST API with Flask, PostgreSQL DB, React frontend, regional languages & SendGrid/Twilio notifications.")
