
# Research Paper Summarizer App (PubMed ID → Summary)

Researchers often face a flood of new publications, making it hard to keep up. An automated summarizer can help by turning a paper’s PubMed ID into a concise, human-readable summary. Summarization is defined as generating a shorter version of a document that preserves its key information. Our goal is to build a Streamlit app: the user types a PubMed ID, and the app fetches the corresponding abstract and uses an open-source transformer to generate a summary. By using free models (no paid APIs) and a simple Python-based web UI, this tool can save researchers’ time and be easily maintained.

## Data Source of Abstracts

We start with a large corpus of biomedical abstracts. For example, Kaggle provides a dataset of \~200K medical-paper abstracts (each tagged with a PubMed ID) for NLP tasks. Similarly, Hugging Face’s **`ccdv/pubmed-summarization`** dataset contains about 133K examples of PubMed articles paired with their abstracts. These datasets illustrate the scale of available data: on the order of 10<sup>5</sup> papers. We use this data in two ways: first, as a lookup table (so we can directly retrieve an abstract given its ID), and second as a source for training or fine-tuning our summarization model if desired. In practice, given a PubMed ID the app either loads the abstract from our local dataset or queries PubMed directly.

## Retrieving Paper Text by PubMed ID

When the user enters a PubMed ID, the app must fetch the paper’s text (typically the abstract). We can do this via NCBI’s API. For example, Python libraries like **Biopython (Entrez)** or the **PyMed** package provide convenient functions to query PubMed by ID. In fact, the EBARA research assistant uses PyMed (“a Python client for querying the PubMed API”) to download abstracts. Our app works the same way: the backend code calls Entrez or PyMed to get the abstract text from NCBI given the PMID. This ensures we always summarize the up-to-date abstract without manual downloads.

## Summarization Model and Technique

We use a free transformer model via Hugging Face for text summarization. Summarization is typically framed as a sequence-to-sequence task, where an encoder-decoder model condenses the input text. A good off-the-shelf choice is **BART** (Bidirectional and Auto-Regressive Transformer). For instance, the `facebook/bart-large-cnn` model is pretrained for summarization and works well on scientific text. We load it using Hugging Face’s pipeline, e.g. `pipeline("summarization", model="facebook/bart-large-cnn")`. Since abstracts may exceed 1024 tokens, we split the text into chunks (with overlap) and summarize each chunk, then optionally concatenate and re-summarize the result. This “hierarchical” approach (summarize summaries) was shown to produce coherent short summaries.

For domain specificity, we can also consider models fine-tuned on medical literature. For example, Hugging Face hosts a “T5 Large for Medical Text Summarization” model, explicitly trained to “generate concise and coherent summaries of medical documents, research papers”. Such a model captures biomedical terminology and style. All of these models are open-source (Apache or similar license), so we can use them freely in our app. The pipeline simply takes the abstract text and returns a summary string.

## Implementation Steps (Pipeline)

1. **Fetch Abstract:** Given the input PubMed ID, retrieve the abstract text via the PubMed API (Entrez/PyMed) or by looking it up in the downloaded dataset.
2. **Preprocess Text:** Clean or truncate the text if needed. If the abstract is very long, split it into manageable chunks (e.g. 1024 tokens) for the summarizer.
3. **Generate Summary:** Run the chunks through the Hugging Face summarization pipeline (BART or a fine-tuned model). Collect and concatenate the partial summaries. Optionally, run a second pass to tighten the combined summary.
4. **Return Output:** Present the final summary in the app’s interface. Ensure it is concise (on the order of 1–3 paragraphs) and captures the paper’s main findings.

Each step can be coded in Python and tested interactively in a notebook. For example, a Colab notebook can be used to prototype the retrieval and summarization functions before integrating them into the app.

## Tools and Environment

* **Google Colab / Jupyter:** We develop and test the code in a Python notebook. Google Colab provides a free cloud-hosted Jupyter environment with no setup and free GPU/TPU access. We can upload the Kaggle dataset to Colab or import it directly.
* **Programming & Models:** We use Python 3 along with the Hugging Face *Transformers* library. Transformers provides high-level pipelines for summarization. For example, BART is loaded as in \[26] and produces summaries out-of-the-box. Models like BART or T5 are free to use.
* **Streamlit:** We build the web interface using Streamlit, an open-source Python framework for creating interactive apps with minimal code. Streamlit lets us define widgets (a text input for the PubMed ID and a “Submit” button) and dynamically display the summary. It handles the web server and re-renders outputs whenever the user submits. This makes the frontend very simple to write.
* **VS Code & GitHub:** For editing and version control, we recommend Visual Studio Code and a GitHub repo. All code (data loading, summarization logic, Streamlit app) is managed in a repo so others can contribute. The Kaggle dataset can be stored in the repo (if size permits) or accessed at runtime. GitHub also allows easy deployment of the Streamlit app via services like Streamlit Cloud.

## App Architecture

*Figure 1: Example layered architecture of the summarizer app.* The *presentation layer* (left) is the Streamlit web UI, where the user enters a PubMed ID. The *business-logic layer* (center) is Python code that fetches the abstract (via PubMed API, e.g. using PyMed) and then calls the summarization model. The *data layer* (right) could be a lightweight database (e.g. SQLite) or cached storage for abstracts and summaries. This separation of concerns makes the system modular: the UI, the retrieval/summarization logic, and the data storage each have distinct roles【27†】.

## Demo Flow

In the deployed app, the user interface shows a text box for the PubMed ID. After entering an ID and clicking **Submit**, the backend pipeline runs automatically. It calls PubMed to get the abstract, splits it if necessary, and feeds it to the summarizer. The resulting summary is displayed on the page (e.g. `st.write(summary)`). For instance, if the user entered an ID for the original Transformer paper (“Attention Is All You Need”), the app might output:

> *“The Transformer is a novel neural network architecture that relies entirely on self-attention mechanisms. It achieves state-of-the-art translation quality after relatively short training and simplifies parallelization. In experiments, the Transformer outperformed previous models on English-to-German and English-to-French tasks.”*

This example (adapted from \[26]) shows how the summary highlights the key contributions. In general, the app’s summary will be much shorter than the original abstract but still informative. The whole process – from ID input to summary output – happens in real time, thanks to the efficient Python and transformer pipeline. Researchers can use this tool to quickly grasp a paper’s main points without manual reading, all powered by free and open-source components.

**Tools Summary:** We rely on Google Colab for development, Python and Hugging Face Transformers for the model, Streamlit for the UI, and standard tools like VS Code and GitHub for coding. This stack ensures the app can be fully implemented with open-source technology, offering a scalable and maintainable solution.

**Sources:** We based our design on best practices from the literature. For instance, Hugging Face’s documentation defines summarization and provides pipelines. Prior work has shown that BART effectively summarizes scientific text, and specialized models (e.g. T5-large for medical text) exist to boost performance on biomedical content. Streamlit’s ease-of-use for Python apps is documented by its creators. Finally, the EBARA project is a close reference: it uses PyMed for PubMed retrieval and a Streamlit frontend, confirming our architecture choices. These sources guided our comprehensive plan for the research paper summarizer.
