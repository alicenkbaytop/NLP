import streamlit as st
from datetime import datetime
from sql_generator import generate_sql
from sql_executor import execute_sql
from shapely import wkt

def render_prompt_input():
    st.subheader("💬 Ask Your Question")
    return st.text_area("Enter your question:", height=100)

def render_buttons(prompt):
    col1, col2 = st.columns([1, 1])

    if col1.button("🚀 Execute Query"):
        if prompt.strip():
            st.markdown("#### 🧾 Generated SQL")
            sql = generate_sql(prompt)
            if sql:
                st.session_state.generated_sql = sql
                df = execute_sql(sql)
                if df is not None:
                    st.session_state.last_df = df

                    st.session_state.query_history.append({
                        "prompt": prompt,
                        "sql": sql,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "rows": len(df)
                    })
        else:
            st.warning("Please enter a question.")

    if col2.button("🗑️ Clear All"):
        st.session_state.generated_sql = ""
        st.session_state.query_history = []
        st.session_state.connection_status = None
        st.session_state.last_df = None
        st.rerun()


def render_results():
    if "last_df" in st.session_state:
        df = st.session_state.last_df
        st.subheader("📊 Query Results")
        if 'SHAPE' in df.columns:
            df = df.drop(columns=['SHAPE'])
        st.dataframe(df)
        
def extract_point_coords(wkt_str):
    try:
        geom = wkt.loads(wkt_str)
        if geom.geom_type == "Point":
            return geom.y, geom.x
    except:
        return None, None
    return None, None

def render_query_history(prompt):
    if st.session_state.query_history:
        st.subheader("📜 Query History")
        for q in reversed(st.session_state.query_history[-10:]):
            st.markdown(f"**Timestamp:** `{q['time']}` | **Rows:** `{q['rows']}`")
            st.markdown(f"**Prompt:** `{q['prompt']}`")
            st.code(q["sql"], language="sql")
