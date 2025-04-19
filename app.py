import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import re

# ---------------------------
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('UK_sponsors.csv')
    df['Organisation Name'] = df['Organisation Name'].astype(str)
    return df

# Load model
@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(df['Organisation Name'].tolist(), convert_to_tensor=True)
    return model, embeddings

df = load_data()
model, company_embeddings = load_model_and_embeddings(df)

# ---------------------------
# UI Header
st.image("https://upload.wikimedia.org/wikipedia/en/a/ae/Flag_of_the_United_Kingdom.svg", width=100)
st.title("UK Skilled Worker Visa Sponsor Checker")

st.markdown("""
Welcome! 👋  
Use this app to find UK companies that sponsor skilled worker visas.  
Just type a **company name** or upload your **CV**, and we’ll show companies that match your field.
""")

# ---------------------------
# Visa Info Panel
with st.expander("📘 Skilled Worker Visa Overview"):
    st.markdown("""
**Key things to know:**
- Your job must be on the [Skilled Worker Occupation List](https://www.gov.uk/government/publications/skilled-worker-visa-immigration-salary-list/skilled-worker-visa-immigration-salary-list)
- It must meet the **minimum salary threshold**
- Exceptions exist for **graduates**, **shortage roles**, and **healthcare workers**

🔗 Learn more:
- [Your job requirements](https://www.gov.uk/skilled-worker-visa/your-job)  
- [When you can be paid less](https://www.gov.uk/skilled-worker-visa/when-you-can-be-paid-less)
""")

# ---------------------------
# CV Upload & Extraction
st.markdown("### 📄 Upload Your CV (Optional)")

uploaded_file = st.file_uploader("Upload a CV or job description (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

cv_text = ""
cv_keywords = ""

def extract_job_keywords(text):
    lines = text.split('\n')
    keywords = []
    for line in lines:
        line = line.strip()
        if len(line.split()) <= 6 and any(word in line.lower() for word in ['assistant', 'engineer', 'developer', 'care', 'nurse', 'manager', 'officer', 'teacher', 'consultant', 'analyst']):
            keywords.append(line)
    return ", ".join(keywords[:3])  # Return top 3 likely roles

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                cv_text += text + "\n"

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            cv_text += para.text + "\n"

    elif uploaded_file.type == "text/plain":
        cv_text = uploaded_file.read().decode("utf-8")

    st.success("✅ CV uploaded successfully!")

    # Show preview
    st.markdown("**CV Preview (first 500 characters):**")
    st.write(cv_text[:500] + "..." if len(cv_text) > 500 else cv_text)

    # Extract keywords
    cv_keywords = extract_job_keywords(cv_text)
    if cv_keywords:
        st.markdown(f"**📌 Extracted Keywords from CV:** `{cv_keywords}`")
    else:
        st.warning("⚠️ No job-related keywords found. Try uploading a clearer CV.")

# ---------------------------
# Search Logic (typed or CV-based)
st.markdown("### 🔍 Search by typing OR by CV")

query = st.text_input("Enter a company name or description:")

search_input = query if query else cv_keywords

if search_input:
    query_embedding = model.encode(search_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, company_embeddings)[0]
    top_results = cos_scores.argsort(descending=True)[:20]

    st.subheader("Top Matches Based on Your Input:")
    for idx in top_results:
        row = df.iloc[int(idx)]
        st.markdown(f"""
        **Organisation Name:** {row['Organisation Name']}  
        **Town/City:** {row['Town/City'] if pd.notna(row['Town/City']) else 'N/A'}  
        **Type & Rating:** `{row['Type & Rating']}`  
        **Route:** `{row['Route']}`  
        """)
else:
    st.info("💡 You can type something above or upload your CV to see results.")

# ---------------------------
# Footer
st.markdown("""
<hr style="margin-top: 40px;">
<p style='text-align: center; font-size: 14px; color: gray;'>
    🚀 Powered by <strong>Viriledigital</strong> | Built with ❤️ using Streamlit & BERT
</p>
""", unsafe_allow_html=True)
