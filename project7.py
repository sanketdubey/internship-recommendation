import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional advanced packages (not required)
try:
    from sentence_transformers import SentenceTransformer, util
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    use_semantic = True
except Exception:
    use_semantic = False

try:
    from googletrans import Translator
    translator = Translator()
    use_translate = True
except Exception:
    use_translate = False

# Try import SendGrid & Twilio
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    has_sendgrid = True
except Exception:
    has_sendgrid = False

try:
    from twilio.rest import Client as TwilioClient
    has_twilio = True
except Exception:
    has_twilio = False

# --------------------------
# UI TEXTS IN DIFFERENT LANGUAGES
# --------------------------
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
        "result": "Top Internship Recommendations for",
        "send_email": "Send Recommendations by Email",
        "send_sms": "Send Recommendations by SMS",
        "email_field": "Recipient Email",
        "phone_field": "Recipient Phone (+91...)"
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
        "result": "के लिए शीर्ष इंटर्नशिप अनुशंसाएं",
        "send_email": "ईमेल द्वारा अनुशंसाएँ भेजें",
        "send_sms": "एसएमएस द्वारा अनुशंसाएँ भेजें",
        "email_field": "प्राप्तकर्ता ईमेल",
        "phone_field": "प्राप्तकर्ता फोन (+91...)"
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
        "result": "साठी शीर्ष इंटर्नशिप शिफारसी",
        "send_email": "ईमेलद्वारे शिफारसी पाठवा",
        "send_sms": "एसएमएसद्वारे शिफारसी पाठवा",
        "email_field": "प्राप्तकर्ता ईमेल",
        "phone_field": "प्राप्तकर्ता फोन (+91...)"
    }
}

# --------------------------
# Load internship data
# --------------------------
@st.cache_data
def load_data():
    # Ensure internships1.csv exists in same folder
    df = pd.read_csv("internships1.csv")
    df['combined'] = df['RequiredSkills'].astype(str) + " " + df['Sector'].astype(str) + " " + df['Location'].astype(str)
    return df

internships = load_data()

# --------------------------
# Page config & top UI
# --------------------------
st.set_page_config(page_title="InternMatch AI", layout="wide")
lang = st.selectbox("Choose Language / भाषा निवडा", ["English", "Hindi", "Marathi"])
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
.small-muted { color:#333; margin:0; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Input form (includes recipient info for notifications)
# --------------------------
with st.form("student_form"):
    name = st.text_input(text['name'])
    education = st.text_input(text['education'])
    skills = st.text_area(text['skills'])
    sector_pref = st.text_input(text['sector'])
    location_pref = st.text_input(text['location'])
    # recipient contact inputs (for demo)
    recipient_email = st.text_input(text['email_field'])
    recipient_phone = st.text_input(text['phone_field'])
    submitted = st.form_submit_button(text['submit'])

# ---------- helper: map language tag to translator codes ----------
lang_code_map = {"English": "en", "Hindi": "hi", "Marathi": "mr"}

# --------------------------
# If form submitted -> compute recommendations
# --------------------------
if submitted:
    student_text = f"{skills} {sector_pref} {location_pref}".strip()

    # translate to English for matching if translator available
    if use_translate and lang != "English":
        try:
            student_text = translator.translate(student_text, src=lang_code_map[lang], dest='en').text
        except Exception as e:
            st.warning(f"Translation failed, proceeding with raw input. ({e})")

    # Use semantic embeddings if available (better)
    if use_semantic:
        internship_embeddings = semantic_model.encode(internships['combined'].tolist(), convert_to_tensor=True)
        student_embedding = semantic_model.encode(student_text, convert_to_tensor=True)
        scores = util.cos_sim(student_embedding, internship_embeddings)[0].cpu().numpy()
    else:
        # TF-IDF fallback
        vectorizer = TfidfVectorizer()
        internship_vectors = vectorizer.fit_transform(internships['combined'])
        student_vector = vectorizer.transform([student_text])
        scores = cosine_similarity(student_vector, internship_vectors)[0]

    internships['score'] = scores
    top3 = internships.sort_values(by='score', ascending=False).head(3).reset_index(drop=True)

    # Save last recommendations in session state for use by buttons
    st.session_state['last_recs'] = top3

    st.subheader(f"{text['result']} {name}:")
    for i, row in top3.iterrows():
        st.markdown(
            f"""
            <div class='reco-card'>
                <div class='reco-title'>{row['Title']}</div>
                <p class='small-muted'><b>Skills Required:</b> {row['RequiredSkills']}</p>
                <p class='small-muted'><b>Sector:</b> {row['Sector']} | <b>Location:</b> {row['Location']}</p>
                <p class='small-muted'><b>Match Score:</b> {round(row['score']*100,2)}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Notifications")
    col1, col2 = st.columns(2)

    # --------------------------
    # Send Email (SendGrid)
    # --------------------------
    with col1:
        if has_sendgrid:
            sg_key = os.environ.get("SENDGRID_API_KEY", "")
            if not sg_key:
                st.warning("SendGrid API key not found in environment. Add SENDGRID_API_KEY or enter below for quick test.")
                sg_key = st.text_input("SendGrid API Key (or set SENDGRID_API_KEY env)", type="password", key="sg_key_input")
            if st.button(text['send_email']):
                if not sg_key:
                    st.error("Provide SendGrid API key to send email.")
                elif not recipient_email:
                    st.error("Please enter recipient email.")
                else:
                    # build email content
                    lines = []
                    for _, r in top3.iterrows():
                        lines.append(f"{r['Title']} | {r['Sector']} | {r['Location']} | Match: {round(r['score']*100,2)}%")
                    email_body = "Your Top Internship Recommendations:\n\n" + "\n".join(lines)
                    try:
                        message = Mail(
                            from_email=os.environ.get("SENDGRID_FROM_EMAIL", "no-reply@example.com"),
                            to_emails=recipient_email,
                            subject="Your Internship Recommendations",
                            plain_text_content=email_body
                        )
                        sg = SendGridAPIClient(sg_key)
                        response = sg.send(message)
                        if response.status_code in (200, 202):
                            st.success("✅ Email sent successfully (via SendGrid).")
                        else:
                            st.error(f"SendGrid returned status {response.status_code}")
                    except Exception as e:
                        st.error(f"Error sending email: {e}")
        else:
            st.info("SendGrid package not installed. Install `sendgrid` to enable email notifications.")

    # --------------------------
    # Send SMS (Twilio)
    # --------------------------
    with col2:
        if has_twilio:
            tw_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
            tw_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
            tw_from = os.environ.get("TWILIO_PHONE_NUMBER", "")  # Twilio trial number like +1XXX or +91XXX

            if not (tw_sid and tw_token):
                st.warning("Twilio SID/Auth not found in env. You can paste keys below for quick test.")
                tw_sid = st.text_input("Twilio SID", key="tw_sid")
                tw_token = st.text_input("Twilio Auth Token", type="password", key="tw_token")
            if not tw_from:
                tw_from = st.text_input("Twilio From Number (your Twilio number, e.g. +1xxx)", key="tw_from")

            if st.button(text['send_sms']):
                if not (tw_sid and tw_token and tw_from):
                    st.error("Provide Twilio SID/Auth and Twilio From Number.")
                elif not recipient_phone:
                    st.error("Enter recipient phone number.")
                else:
                    try:
                        client = TwilioClient(tw_sid, tw_token)
                        sms_body = "Your Top Internship Recommendations:\n"
                        for _, r in top3.iterrows():
                            sms_body += f"{r['Title']} ({r['Location']}) Match: {round(r['score']*100,2)}% ; "
                        # Twilio trial accounts can only send to verified numbers
                        message = client.messages.create(body=sms_body, from_=tw_from, to=recipient_phone)
                        st.success("✅ SMS sent successfully (via Twilio). SID: " + (message.sid or ""))
                    except Exception as e:
                        st.error(f"Error sending SMS: {e}")
        else:
            st.info("Twilio package not installed. Install `twilio` to enable SMS notifications.")

# Footer info
st.info(
    "This prototype sends notifications via SendGrid (email) and Twilio (SMS). "
    "Use environment variables SENDGRID_API_KEY, SENDGRID_FROM_EMAIL, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER "
    "or paste keys in the UI for quick testing. Trial accounts may require verified recipient numbers."
)
