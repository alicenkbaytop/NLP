import streamlit as st
from config import setup_page
from state import initialize_session_state
from sidebar import render_sidebar
from components import render_prompt_input, render_buttons, render_results, render_query_history

setup_page()
initialize_session_state()
render_sidebar()

# Prompt input area
prompt = render_prompt_input()

# Button logic and results
render_buttons(prompt)

# Display results and history
render_results()
render_query_history(prompt)
