import streamlit as st
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Streamlit app title
st.title("README.md Generator")

# Input for the URL to scrape
url = st.text_input("Enter the URL to scrape:")

# Button to trigger the scraping and README generation
if st.button("Generate README.md"):
    if url:
        FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
        
        # Initialize FireCrawlLoader with your API key and the URL to scrape
        loader = FireCrawlLoader(
            api_key=FIRECRAWL_API_KEY, url=url, mode="scrape"
        )

        # Load the data from the webpage
        data = loader.load()

        # Initialize ChatOllama for processing the scraped data
        model = "deepseek-r1"
        temperature = 0
        llm = ChatOllama(model=model, temperature=temperature)

        # Define the prompt for generating a README
        prompt_markdown = f"""
        Generate a `README.md` file based on the scraped webpage content.

        Scraped Content:
        {data}

        Return only the markdown content using the structure below:
        # Project Name

        ## Description

        ## Getting Started

        ## Dependencies

        ## Installation

        ## Usage

        ## Help

        ## Authors

        ## Version History

        ## License

        ## Acknowledgments

        Strictly return only the markdown content, without any additional text, explanations, or formatting outside the specified structure. Do not include any thinking steps, notes, or extra commentary.
        """

        # Generate the README content
        response = llm.invoke(prompt_markdown)

        # Extract the content from the AIMessage object
        response_content = response.content

        # Post-process the response to remove unwanted content
        # Remove everything before the first markdown header (#)
        cleaned_content = response_content[response_content.find("```"):]

        # Display the cleaned README content in a text area for preview
        st.subheader("Preview README.md")
        st.text_area("README.md", cleaned_content, height=400)

        # Provide a download button for the README.md file
        st.download_button(
            label="Download README.md",
            data=cleaned_content,
            file_name="README.md",
            mime="text/markdown"
        )
    else:
        st.error("Please enter a valid URL.")