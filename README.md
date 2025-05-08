#  SHL Assessment Recommendation System

An AI-powered recommendation system designed to suggest the most relevant SHL assessments based on user queries, job descriptions, or unstructured input data.  
The system combines **Natural Language Processing (NLP)** techniques and **Large Language Models (LLMs)** to deliver accurate and context-aware recommendations through an interactive frontend.

---

##  Features

- **Assessment Recommendations** based on:
  - Job descriptions
  - Unstructured URLs or text input
  - Custom user queries

- **NLP-Powered Matching**
  - Semantic similarity using **Sentence-BERT embeddings**
  - Context extraction and filtering with **Gemini 1.5 Pro (LLM)**

- **Smart Ranking & Scoring**
  - Cosine similarity-based ranking
  - Relevance filtering for top recommendations

- **User-Friendly Frontend**
  - Built with **Streamlit** for easy interaction

---

##  Tech Stack

| Component             | Tool / Library                 |
|-----------------------|--------------------------------|
| NLP Embeddings        | Sentence-BERT (`all-MiniLM-L6-v2`) |
| LLM Integration       | Gemini 1.5 Pro (Google Generative AI) |
| Frontend Interface    | Streamlit                      |
| Backend API           | FastAPI                        |
| Data Manipulation     | Pandas                         |
| Similarity Scoring    | Cosine Similarity (PyTorch)    |

---

##  How it Works

1. **Input Handling**  
   The user enters a query, JD text, or JD URL.

2. **Feature Extraction (LLM)**  
   The system uses **Gemini 1.5 Pro** to extract structured features such as job role, skills, duration, etc.

3. **Semantic Search (NLP)**  
   Both the user query and catalog assessments are embedded using **Sentence-BERT**.  
   Cosine similarity is computed to find top-matching assessments.

4. **Intelligent Filtering (LLM)**  
   Gemini further filters recommendations based on user constraints (e.g., duration, required skills).

5. **Recommendation Output**  
   The top relevant assessments are displayed via the Streamlit interface.

---

##  Performance Metrics

| Model Type          | Recall@5 | MAP@5 |
|---------------------|----------|-------|
| NLP Model Only      | 0.85     | 0.71  |
| NLP + LLM Hybrid    | 1.00     | 1.00  |

---

##  Example Test Cases

> These sample queries demonstrate how our system delivers precise recommendations:

- *"Hiring Java developers with collaboration skills, assessment duration under 40 mins"*
- *"Looking for mid-level Python, SQL, JavaScript assessments within 60 mins"*
- *"Need cognitive and personality tests for analyst roles, max 45 mins"*
- *[LinkedIn JD URL: SHL Research Engineer AI Job Description](https://www.linkedin.com/jobs/view/research-engineer-ai-at-shl-4194768899/?originalSubdomain=in)*

---

##  Data Source

The SHL assessment catalog data was collected from the official SHL product catalog:  
[SHL Product Catalog](https://www.shl.com/solutions/products/product-catalog/)

---

##  Authors

> This system was built as part of the SHL Research Internship take-home assessment.

---

##  Final Notes

This project showcases the power of combining **semantic search** and **LLM-based understanding** to make hiring assessments selection smarter and faster.

