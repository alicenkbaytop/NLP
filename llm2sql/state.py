import streamlit as st

def initialize_session_state():
    defaults = {
        "generated_sql": "",
        "query_history": [],
        "connection_status": None,
        "dsn": "",
        "username": "VERIBILIMI",
        "password": "123456",
        "connection_method": "TNS Names",
        "ollama_model": "qwen3",
        "timeout_seconds": 60,
        "tns_admin_path": r"C:\app\client\cbs\product\19.0.0\client_x64\Network\Admin",
        "table_name": "CBS.POI",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
