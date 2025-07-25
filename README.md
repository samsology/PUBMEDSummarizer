# 🧠 PubMed Research Paper Summarizer

A simple and elegant web app that summarizes biomedical research abstracts based on PubMed IDs using a pretrained transformer model (BART). Designed to help students, researchers, and medical professionals get concise insights without reading entire abstracts.

## 🚀 Demo

Try it here: [https://researchpapersummarizer.streamlit.app](https://pubmedsummarizer.streamlit.app)

## 📁 Dataset

We used a curated dataset of **200,000+ PubMed abstracts** available from Kaggle:

🔗 [200k PubMed Abstracts Dataset](https://www.kaggle.com/datasets/anshulmehtakaggl/200000-abstracts-for-seq-sentence-classification)

> ⚠️ **Note:** This app only supports PubMed IDs that exist in the dataset above. We are working on integrating a live API to support all PubMed articles in the future.

---

## ✨ Features

- 🔍 **Search by PubMed ID**
- 🤖 **Summarize abstracts using BART (via Hugging Face)**
- 💡 **Fast and intuitive UI built with Streamlit**
- 🛠️ **Offline summarization using a preloaded CSV dataset**

---

## 🛠️ Tech Stack

- Python
- Pandas
- Hugging Face Transformers (BART)
- Google Colab (for development and testing)
- Streamlit (UI)
- GitHub (version control)

---

## 📦 Installation

To run this project locally:

1. **Clone the repository**

git clone https://github.com/JohnsonSamuel/pubmedsummarizer.git
cd researchpapersummarizer

🔧 Project Structure

researchpapersummarizer/
│
├── app.py                  # Main Streamlit application
├── summarize.py           # Core logic: loading data and generating summaries
├── abstracts.csv          # Preprocessed dataset (200k PubMed abstracts)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore


🤝 Team
1. Johnson Samuel – https://www.linkedin.com/in/samuel-johnson-766b2a337/
2. Kuburah Otaru – https://www.linkedin.com/in/kuburat-otaru/

📌 Roadmap
 Build prototype with offline dataset

 Integrate Hugging Face summarizer (BART)

 Deploy with Streamlit

 Add live PubMed API support

 Enable batch summarization

 Add keyword/tag extraction feature

 📢 Acknowledgments
 
Hugging Face 🤗 for the BART model

Streamlit for simple UI development

Kaggle for the open-access abstract dataset
 

