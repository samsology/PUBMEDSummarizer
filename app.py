import streamlit as st
from summarize import load_abstracts, summarize_by_pmid

st.set_page_config(page_title="Research Paper Summarizer", layout="centered")

st.title("🧠 Research Paper Summarizer")
st.write("Enter a **PubMed ID** to generate a summary of its abstract.")

# Load abstracts (cached)
@st.cache_data
def load_data():
    return load_abstracts("papers.txt")

df = load_data()

# Input field
pmid = st.text_input("🔍 PubMed ID:", "")

if st.button("Summarize"):
    if pmid:
        with st.spinner("Generating summary..."):
            summary = summarize_by_pmid(df, pmid.strip())
            if summary:
                st.success("✅ Summary generated!")
                st.markdown(f"**Summary of PubMed ID {pmid}:**")
                st.write(summary)
            else:
                st.error(f"❌ No abstract found for PubMed ID: {pmid}")
    else:
        st.warning("Please enter a valid PubMed ID.")
