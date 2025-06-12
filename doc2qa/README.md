# 📄 Document QA Assistant

A Streamlit web application that allows users to upload documents (PDF, images, DOCX), extracts the text content, and enables users to ask questions about the document using the **Gemma 2 9B Instruct model** hosted on **Groq**.

---

## 🚀 Features

- ✅ Supports PDF, PNG, JPG, and DOCX file uploads
- ✅ Extracts text using OCR (for images), PDF parsing, and DOCX reading
- ✅ Asks natural language questions about the uploaded content
- ✅ Uses `Gemma-2-9b-it` via **Groq API** for smart, fast answers
- ✅ Simple and user-friendly UI with Streamlit

---

### git clone 
### 📚 Python dependencies

Install them with:

```bash
pip install -r requirements.txt
```

### Create .env file with your Groq API key:
* GROQ_API_KEY

### Run the app:
* streamlit run app.py