import pandas as pd
from transformers import pipeline

# Load and parse dataset
def load_abstracts(file_path='papers.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    current_id = None
    current_abstract = []

    for line in lines:
        line = line.strip()
        if line.startswith('###'):
            if current_id and current_abstract:
                records.append({
                    'PMID': current_id,
                    'Abstract': ' '.join(current_abstract)
                })
            current_id = line.replace('###', '').strip()
            current_abstract = []
        elif '\t' in line:
            parts = line.split('\t', 1)
            if len(parts) == 2:
                current_abstract.append(parts[1])
    
    if current_id and current_abstract:
        records.append({
            'PMID': current_id,
            'Abstract': ' '.join(current_abstract)
        })

    return pd.DataFrame(records)

# Load model once
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Summarize given PubMed ID
def summarize_by_pmid(df, pmid):
    paper = df[df['PMID'] == str(pmid)]
    if paper.empty:
        return None

    text = paper.iloc[0]['Abstract'][:1024]  # Truncate
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    return summary
