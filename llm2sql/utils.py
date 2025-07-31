import re

def clean_sql(sql_text):
    # Remove any Markdown or extra info if present
    match = re.search(r"```sql\s*(.*?)```", sql_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        sql_text = match.group(1)
    else:
        # Try to extract the last line that looks like SQL
        lines = sql_text.strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line.upper().startswith("SELECT"):
                sql_text = line
                break

    # Remove non-printable characters and BOM
    sql_text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", sql_text)
    sql_text = sql_text.replace("\ufeff", "")

    # Fix missing space around FROM
    sql_text = re.sub(r"\*\s*FROM", "* FROM", sql_text, flags=re.IGNORECASE)

    # Fix missing space before WHERE, AND, etc.
    sql_text = re.sub(
        r"([a-zA-Z0-9_])\s*(WHERE|AND|OR|GROUP BY|ORDER BY|HAVING|JOIN)\b",
        r"\1 \2",
        sql_text,
        flags=re.IGNORECASE,
    )

    # Add space after commas
    sql_text = re.sub(r",\s*", ", ", sql_text)

    # Remove trailing semicolon
    return sql_text.strip().rstrip(";")
