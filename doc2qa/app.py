import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import docx
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Helper: Extract text from different file types
def extract_text(file):
    file_type = file.type
    if file_type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file_type in ["image/png", "image/jpeg"]:
        return extract_text_from_image(file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    else:
        return "Unsupported file format."

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Helper: Call Groq API with Gemma-2-9b-it model
def query_gemma(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "gemma2-9b-it",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided document."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Streamlit UI
st.title("📄 Doc2QA Assistant")
uploaded_file = st.file_uploader("Upload a document (PDF, PNG, JPG, DOCX)", type=["pdf", "png", "jpg", "jpeg", "docx"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        document_text = extract_text(uploaded_file)
    
    if document_text.strip():
        st.success("Text extracted successfully!")
        st.text_area("Extracted Text", document_text, height=200)
        
        question = st.text_input("**Ask a question about the document:**")
        if question:
            with st.spinner("Thinking..."):
                full_prompt = f"""Based on the following document:\n\n{document_text}\n\nAnswer the following question:\n{question}"""
                try:
                    answer = query_gemma(full_prompt)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error querying model: {e}")
    else:
        st.warning("No text could be extracted from the file.")
