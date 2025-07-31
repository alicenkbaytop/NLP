import streamlit as st
from database import test_connection

def render_sidebar():
    with st.sidebar:
        st.header("🔐 Database Configuration")

        username = st.text_input("Username", value=st.session_state.username)
        password = st.text_input("Password", type="password", value=st.session_state.password)
        method = st.selectbox("Connection Method", ["TNS Names", "Easy Connect", "Connection String"])

        dsn, tns_admin = "", st.session_state.tns_admin_path
        if method == "Easy Connect":
            host = st.text_input("Host", value="localhost")
            port = st.text_input("Port", value="1521")
            service_name = st.text_input("Service Name", value="PUBCBS")
            dsn = f"{host}:{port}/{service_name}"
        elif method == "TNS Names":
            dsn = st.text_input("TNS Name (DSN)", value="PUBCBS")
            tns_admin = st.text_input("TNS_ADMIN Path", value=tns_admin)
        else:
            dsn = st.text_area("Connection String", value="")

        table_name = st.text_input("Table Name", value=st.session_state.get("table_name", ""))
        st.session_state["table_name"] = table_name

        st.session_state.update({
            "username": username,
            "password": password,
            "connection_method": method,
            "tns_admin_path": tns_admin,
            "dsn": dsn
        })

        st.session_state.ollama_model = st.selectbox("Ollama Model", ["qwen3"])
        st.session_state.timeout_seconds = st.slider("Query Timeout", 30, 180, st.session_state.timeout_seconds)

        if st.button("🔗 Test Connection"):
            test_connection()
