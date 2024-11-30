from langchain_community.llms import Ollama
import streamlit as st

llm = Ollama(model="llama3")

st.title("Open Source Chatbot")

prompt = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            st.write_stream(llm.stream(prompt, stop=['<|eot_id|>']))