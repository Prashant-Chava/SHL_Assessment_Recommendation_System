import os
import re
import json
import numpy as np
import pandas as pd
import torch
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# --- 1. Load Dataset ---
catalog_df = pd.read_csv("SHL_catalog.csv")

def combine_row(row):
    """Combine multiple columns into a single string for embedding."""
    parts = [
        str(row["Assessment Name"]),
        str(row["Duration"]),
        str(row["Remote Testing Support"]),
        str(row["Adaptive/IRT"]),
        str(row["Test Type"]),
        str(row["Skills"]),
        str(row["Description"]),
    ]
    return ' '.join(parts)

catalog_df['combined'] = catalog_df.apply(combine_row, axis=1)
corpus = catalog_df['combined'].tolist()

# --- 2. Load Embedding Model ---
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# --- 3. Utility Functions ---

def extract_url_from_text(text):
    """Extract first URL from a string."""
    match = re.search(r'(https?://[^\s,]+)', text)
    return match.group(1) if match else None

def extract_text_from_url(url):
    """Extract visible text content from a URL."""
    try:
        response = requests.get(url, headers={'User-Agent': "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join(soup.get_text().split())
    except Exception as e:
        return f"Error: {e}"

def convert_numpy(obj):
    """Ensure Numpy types are JSON serializable."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# --- 4. Configure Gemini API ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# --- 5. Core Functions ---

def extract_features_with_llm(user_query):
    """Use LLM to extract key hiring features from user query or JD."""
    prompt = f"""
You are an intelligent assistant helping to recommend SHL assessments.

The input below may be:
1. A natural language query describing assessment needs
2. A job description (JD)
3. A JD URL (converted to text)
4. A combination of query + JD

Extract these **if available**:
- Job Title  
- Duration  
- Remote Support (Yes/No)  
- Adaptive (Yes/No)  
- Test Type  
- Skills  
- Other relevant info

Format as one line:
`<Job Title> <Duration> <Remote Support> <Adaptive> <Test Type> <Skills> <Other Info>`

Skip fields not mentioned.  
---
Input:  
{user_query}

Return only the final sentence — no explanations.
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def find_assessments(user_query, k=5):
    """Find top-k similar assessments to user query."""
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = min(k, len(corpus))
    top_results = torch.topk(cosine_scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()
        result = {
            "Assessment Name": catalog_df.iloc[idx]['Assessment Name'],
            "Skills": catalog_df.iloc[idx]['Skills'],
            "Test Type": catalog_df.iloc[idx]['Test Type'],
            "Description": catalog_df.iloc[idx]['Description'],
            "Remote Testing Support": catalog_df.iloc[idx]['Remote Testing Support'],
            "Adaptive/IRT": catalog_df.iloc[idx]['Adaptive/IRT'],
            "Duration": catalog_df.iloc[idx]['Duration'],
            "URL": catalog_df.iloc[idx]['URL'],
            "Score": round(score.item(), 4)
        }
        results.append(result)
    return results

def filter_relevant_assessments_with_llm(user_query, top_results):
    """Use LLM to filter most relevant assessments."""
    prompt = f"""
You are refining assessment recommendations based on user needs.

User query: "{user_query}"  
You are given 10 or fewer assessments.

Filter based on:
- Duration match  
- Skills match  
- Remote support, Adaptive, Test type  
- Return **minimum 1**, max 10 (no empty list)

Respond in JSON array:
[
  {{
    "Assessment Name": "...",
    "Skills": "...",
    "Test Type": "...",
    "Description": "...",
    "Remote Testing Support": "...",
    "Adaptive/IRT": "...",
    "Duration": "... mins",
    "URL": "...",
    "Score": ...
  }},
  ...
]

---
Assessments:  
{top_results}
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def query_handling_using_LLM_updated(query):
    """Main handler — process query, fetch and filter recommendations."""
    url = extract_url_from_text(query)

    # Append extracted text if URL is found
    if url:
        extracted_text = extract_text_from_url(url)
        query += " " + extracted_text

    # Extract structured features from query
    user_query = extract_features_with_llm(query)

    # Get top 10 similar assessments
    top_results = find_assessments(user_query, k=10)

    # Convert to JSON for filtering
    top_json = json.dumps(top_results, indent=2, default=convert_numpy)

    # LLM filters relevant ones
    filtered_output = filter_relevant_assessments_with_llm(user_query, top_json)

    # Extract JSON array from LLM response
    try:
        match = re.search(r"\[.*\]", filtered_output, re.DOTALL)
        if match:
            json_str = match.group()
            filtered_results = json.loads(json_str)
        else:
            print("⚠️ No valid JSON array found in the response:")
            print(filtered_output)
            return pd.DataFrame()
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        print("Raw output was:\n", filtered_output)
        return pd.DataFrame()

    # Return as DataFrame
    if filtered_results:
        return pd.DataFrame(filtered_results)
    else:
        return pd.DataFrame()
