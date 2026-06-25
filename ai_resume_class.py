import streamlit as st
import re
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set up page styling
st.set_page_config(page_title="AI Resume Screener", page_icon="🤖", layout="wide")

# ==========================================
# 1. DATA & AI MODEL SETUP
# ==========================================

@st.cache_resource
def train_model():
    # Mock data to train our lightweight classifier
    training_data = {
        'Resume': [
            "Python developer backend skills Django Flask PostgreSQL REST API AWS Docker software engineer Linux",
            "Data Scientist machine learning model training Python pandas scikit-learn deep learning NLP SQL Tableau",
            "Human Resources talent acquisition recruiting payroll onboarding employee relations manager HR metrics",
            "Frontend Engineer React Vue HTML CSS JavaScript UI UX web design responsive layout Webpack",
            "DevOps Engineer Kubernetes Docker Jenkins CI CD pipeline cloud AWS Azure terraform system admin"
        ],
        'Category': [
            'Backend Developer',
            'Data Scientist',
            'Human Resources',
            'Frontend Developer',
            'DevOps Engineer'
        ]
    }
    df = pd.DataFrame(training_data)
    
    # Simple cleaner
    def clean(text):
        return re.sub(r'[^\w\s]', ' ', text.lower()).strip()
        
    df['Cleaned'] = df['Resume'].apply(clean)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Cleaned'])
    y = df['Category']
    
    model = LogisticRegression()
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_pdf_text(uploaded_file):
    text = ""
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================

st.title("🤖 AI Resume Analyzer & Classifier")
st.write("Upload a candidate's resume in PDF format to automatically classify their core domain and parse vital keywords.")

st.markdown("---")

# Layout columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Upload Corner")
    uploaded_file = st.file_uploader("Drop Resume PDF here", type=["pdf"])

with col2:
    st.subheader("📊 Analysis Dashboard")
    
    if uploaded_file is not None:
        # 1. Parse and clean
        with st.spinner("Extracting text and scanning profile..."):
            raw_text = extract_pdf_text(uploaded_file)
            cleaned_text = clean_text(raw_text)
            
        if cleaned_text:
            # 2. Predict Model Category
            vec_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vec_text)[0]
            probs = model.predict_proba(vec_text)[0]
            confidence = max(probs) * 100
            
            # 3. Highlight identified skills
            skill_bank = ["python", "django", "react", "kubernetes", "aws", "machine learning", "recruiting", "sql", "javascript", "docker"]
            found_skills = [skill.upper() for skill in skill_bank if skill in cleaned_text]
            
            # 4. Render Metrics
            st.success("Analysis Complete!")
            
            m1, m2 = st.columns(2)
            m1.metric(label="Predicted Job Role", value=prediction)
            m2.metric(label="Match Confidence", value=f"{confidence:.1f}%")
            
            st.markdown("### 💡 Detected Core Technical Skills")
            if found_skills:
                # Render skills as colorful badges/tags using modern formatting
                skills_html = "".join([f"<span style='background-color:#0073e6;color:white;padding:5px 10px;margin:5px;border-radius:5px;display:inline-block;font-weight:bold;'>{s}</span>" for s in found_skills])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.info("No explicit core keywords from our baseline bank matched.")
                
            # 5. Raw Text Previewer
            with st.expander("📄 View Parsed Clean Text Preview"):
                st.text_area("Extracted Text Tokens", cleaned_text[:1500] + "...", height=250)
        else:
            st.warning("Could not extract legible text characters from this PDF document.")
    else:
        st.info("Awaiting file upload... Drop a PDF on the left panel to begin parsing.")