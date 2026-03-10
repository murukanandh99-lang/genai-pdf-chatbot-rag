import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Analyzer & ATS Score")

resume_file = st.file_uploader("Upload Resume", type="pdf")
job_description = st.text_area("Paste Job Description")

def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

if resume_file and job_description:

    resume_text = extract_text(resume_file)

    documents = [resume_text, job_description]

    cv = CountVectorizer()
    matrix = cv.fit_transform(documents)

    similarity = cosine_similarity(matrix)[0][1]

    ats_score = round(similarity * 100,2)

    st.subheader(f"ATS Score: {ats_score}%")

    if ats_score > 70:
        st.success("Good Resume Match")
    else:
        st.warning("Add more keywords from Job Description")
