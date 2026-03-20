import streamlit as st

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AI Resume Matcher Pro",
    page_icon="📄",
    layout="wide"
)

import PyPDF2
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ FIXED NLTK SETUP (CLOUD SAFE)
@st.cache_resource
def setup_nltk():
    nltk_data_dir = "/tmp/nltk_data"
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords"
    }

    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, download_dir=nltk_data_dir)

# ✅ CALL AFTER CONFIG
setup_nltk()

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main {
    background-color: white;
    color: black;
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🤖 Resume Matcher Pro")
    st.write("Upgrade your resume using AI insights 🚀")

    st.divider()

    st.subheader("📌 Features")
    st.write("""
    - Match Score
    - Keyword Analysis
    - Missing Skills Detection
    - Multi Resume Ranking
    """)

# ---------- FUNCTIONS ----------

def extract_text(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    except:
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def preprocess(text):
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    try:
        words = word_tokenize(text)
    except:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        words = text.split()

    return [w for w in words if w not in stop_words]

def calculate_similarity(resume, job):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([resume, job])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100
    return round(score, 2)

def extract_keywords(text, top_n=15):
    words = preprocess(clean_text(text))
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]

# ---------- MAIN UI ----------

def main():
    st.title("📄 AI Resume Matcher Pro")
    st.caption("Analyze & improve your resume instantly ⚡")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_files = st.file_uploader(
            "📤 Upload Resumes (PDF)",
            type=["pdf"],
            accept_multiple_files=True
        )

    with col2:
        job_description = st.text_area("📝 Paste Job Description", height=250)

    if st.button("🔍 Analyze Resume"):

        if not uploaded_files:
            st.warning("Please upload at least one resume")
            return

        if not job_description:
            st.warning("Please paste job description")
            return

        results = []

        with st.spinner("Analyzing..."):

            job_clean = clean_text(job_description)
            job_keywords = set(extract_keywords(job_description))

            for file in uploaded_files:
                resume_text = extract_text(file)

                if not resume_text:
                    continue

                resume_clean = clean_text(resume_text)
                score = calculate_similarity(resume_clean, job_clean)

                resume_keywords = set(extract_keywords(resume_text))

                matched = resume_keywords & job_keywords
                missing = job_keywords - resume_keywords

                results.append({
                    "name": file.name,
                    "score": score,
                    "matched": matched,
                    "missing": missing
                })

        if not results:
            st.error("No valid resumes processed")
            return

        results.sort(key=lambda x: x["score"], reverse=True)

        st.divider()
        st.subheader("🏆 Resume Ranking")

        for i, r in enumerate(results, 1):
            st.write(f"{i}. {r['name']} — {r['score']}%")

        st.divider()
        st.subheader("📊 Detailed Results")

        for r in results:
            with st.expander(f"📄 {r['name']}"):

                st.metric("Match Score", f"{r['score']}%")
                st.progress(int(r["score"]))

                if r["score"] < 40:
                    st.error("❌ Low Match")
                elif r["score"] < 70:
                    st.warning("⚠️ متوسط Match")
                else:
                    st.success("✅ Excellent Match")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("✅ Matched Keywords")
                    if r["matched"]:
                        st.success(", ".join(list(r["matched"])))
                    else:
                        st.info("No strong matches")

                with col2:
                    st.subheader("❌ Missing Keywords")
                    if r["missing"]:
                        st.error(", ".join(list(r["missing"])))
                    else:
                        st.success("No missing keywords 🎉")

                report = f"""
Resume: {r['name']}
Match Score: {r['score']}%

Matched Keywords:
{', '.join(r['matched'])}

Missing Keywords:
{', '.join(r['missing'])}
"""

                st.download_button(
                    f"📥 Download {r['name']} Report",
                    data=report,
                    file_name=f"{r['name']}_analysis.txt"
                )

# RUN
if __name__ == "__main__":
    main()