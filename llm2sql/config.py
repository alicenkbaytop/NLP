import streamlit as st

def setup_page():
    st.set_page_config(page_title="Chat2SQL", layout="wide", page_icon="")
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stCode {
            background-color: #f0f2f6;
            border-left: 5px solid #667eea;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>Chat2SQL</h1>
        <p>Transform your questions into SQL queries and get instant results</p>
    </div>
    """, unsafe_allow_html=True)
