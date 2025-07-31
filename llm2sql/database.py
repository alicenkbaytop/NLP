import streamlit as st
import os
import oracledb

def test_connection():
    method = st.session_state.connection_method
    if method == "TNS Names":
        path = st.session_state.tns_admin_path
        if not os.path.isdir(path):
            st.error("Invalid TNS_ADMIN path.")
            st.session_state.connection_status = "error"
            return

        try:
            oracledb.init_oracle_client(config_dir=path)
        except Exception as e:
            st.error(f"Oracle client init failed: {e}")
            st.session_state.connection_status = "error"
            return

    try:
        conn = oracledb.connect(
            user=st.session_state.username,
            password=st.session_state.password,
            dsn=st.session_state.dsn
        )
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM DUAL")
        st.success("Connection successful!")
        st.session_state.connection_status = "success"
    except Exception as e:
        st.error(f"Connection failed: {e}")
        st.session_state.connection_status = "error"
