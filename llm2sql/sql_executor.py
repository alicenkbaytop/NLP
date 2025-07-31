import streamlit as st
import pandas as pd
import oracledb

def execute_sql(sql):
    if st.session_state.connection_status != "success":
        st.warning("No active DB connection.")
        return None

    try:
        conn = oracledb.connect(
            user=st.session_state.username,
            password=st.session_state.password,
            dsn=st.session_state.dsn
        )
        conn.callTimeout = st.session_state.timeout_seconds * 1000
        cursor = conn.cursor()
        cursor.execute(sql)

        columns = [col[0] for col in cursor.description]
        data = cursor.fetchmany(1000)
        df = pd.DataFrame(data, columns=columns)
        cursor.close()
        conn.close()

        return df
    except Exception as e:
        st.error(f"Query execution failed: {e}")
        return None
