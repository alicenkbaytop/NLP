import subprocess
import streamlit as st
from utils import clean_sql

def generate_sql(prompt):
    table_name = st.session_state.get("table_name", "").strip()
    # Define static schema for CBS.POI
    poi_schema = """
    You must use only the following table and columns:

    Table: CBS.POI
    Columns:
    - OBJECTID
    - POI_ID
    - POI_ADI
    - BINA_ADI
    - ANA_KATEGORI
    - ALT_KATEGORI
    - ILCE_UAVT
    - ILCE_ADI
    - MAHALLE_UAVT
    - MAHALLE_ADI
    - YOLKN
    - YOL_ADI
    - YAPIKN
    - KAPIKN
    - KAPI_NO
    - CREATED_USER
    - CREATED_DATE
    - LAST_EDITED_USER
    - LAST_EDITED_DATE
    - URETIM_KALITESI
    - DURUM
    - GLOBALID
    - GRUP_ADI
    - HIYERARSI
    - ADRES
    - ONCELIK
    - SEMT_ADI
    - ACIKLAMA
    - BILINEN
    - NEAR
    - VERI_KAYNAGI
    - TAM_ADI
    - Z
    - SE_ANNO_CAD_DATA
    - KONTROL
    """

    table_info = f"The table you should use is named `{table_name}`." if table_name else ""

    context = f"""
        You are an Oracle 11g SQL expert. Convert the request into a valid Oracle SQL query.
        Do not explain. Do not use markdown. Just return the SQL.
        
        {table_info}
        {poi_schema}

        Request: {prompt}

        SQL:
    """

    try:
        result = subprocess.run(
            ["ollama", "run", st.session_state.ollama_model],
            input=context,
            text=True,
            capture_output=True,
            timeout=st.session_state.timeout_seconds,
            encoding="utf-8",
            errors="replace"
        )
        if result.returncode == 0:
            st.code(clean_sql(result.stdout), language="sql")
            return clean_sql(result.stdout)
        else:
            st.error(f"Ollama error: {result.stderr}")
    except Exception as e:
        st.error(f"Ollama error: {e}")
    return None
