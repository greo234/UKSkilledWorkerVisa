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

query = st.text_input("Enter a company name or description (e.g., 'care agency in London'):")

if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, company_embeddings)[0]
    top_results = cos_scores.argsort(descending=True)[:20]

    st.subheader("Top Matches:")
    for idx in top_results:
        row = df.iloc[int(idx)]
        st.markdown(f"""
        **Organisation Name:** {row['Organisation Name']}  
        **Town/City:** {row['Town/City'] if pd.notna(row['Town/City']) else 'N/A'}  
        **Type & Rating:** `{row['Type & Rating']}`  
        **Route:** `{row['Route']}`  
        """)
