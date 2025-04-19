import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('UK_sponsors.csv')
    df['Organisation Name'] = df['Organisation Name'].astype(str)
    return df

# Load model and encode company names
@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(df['Organisation Name'].tolist(), convert_to_tensor=True)
    return model, embeddings

df = load_data()
model, company_embeddings = load_model_and_embeddings(df)

# App UI
st.image("https://upload.wikimedia.org/wikipedia/en/a/ae/Flag_of_the_United_Kingdom.svg", width=100)
st.title("UK Skilled Worker Visa Sponsor Checker")

st.markdown("""
Welcome! 👋  
Use this app to find UK companies that sponsor skilled worker visas.  
Just type in a **company name** or a **description** and
We'll find the closest matches for you.  
""")

with st.expander("📘 Skilled Worker Visa Overview"):
    st.markdown("""
**Want to understand how skilled worker visa sponsorship works?**  
Here are some key points:
- Your job must be on the [Skilled Worker Occupation List](https://www.gov.uk/government/publications/skilled-worker-visa-going-rates-for-eligible-occupations/skilled-worker-visa-going-rates-for-eligible-occupation-codes)
- It must meet the **minimum salary requirement** of at least £38, 700 per year or the 'going rate' for your role, whichever is higher.
- Exceptions exist for **new graduates**, **shortage roles**, and **education/healthcare workers**

🔗 Learn more:  
- [Your job requirements](https://www.gov.uk/skilled-worker-visa/your-job)  
- [When you can be paid less](https://www.gov.uk/skilled-worker-visa/when-you-can-be-paid-less)
    """)


query = st.text_input("Enter a company name or description")

if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, company_embeddings)[0]
    top_results = cos_scores.argsort(descending=True)[:10]

    st.subheader("Top Matches:")
    for idx in top_results:
        row = df.iloc[int(idx)]
        st.markdown(f"""
        **Organisation Name:** {row['Organisation Name']}  
        **Town/City:** {row['Town/City'] if pd.notna(row['Town/City']) else 'N/A'}  
        **Type & Rating:** `{row['Type & Rating']}`  
        **Route:** `{row['Route']}`  
        """)
st.markdown("""
<hr style="margin-top: 40px;">
<p style='text-align: center; font-size: 14px; color: gray;'>
    🚀 Powered by <strong>Viriledigital</strong> | Built with ❤️ using Streamlit & BERT
</p>
""", unsafe_allow_html=True)
