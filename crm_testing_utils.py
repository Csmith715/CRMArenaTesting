# --- soql_to_sql.py ---
import re

def soql_to_sql(soql: str) -> str:
    s = soql.strip().rstrip(";")

    # SELECT COUNT() FROM Object
    m = re.fullmatch(r"SELECT\s+COUNT\(\)\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*)", s, re.I)
    if m:
        obj = m.group(1)
        return f"SELECT COUNT(*) AS count FROM {obj}"

    # SELECT fields FROM Object WHERE <simple predicates>
    m = re.fullmatch(
        r"SELECT\s+(.+?)\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+WHERE\s+(.+))?",
        s, re.I
    )
    if m:
        fields = m.group(1).strip()
        table = m.group(2).strip()
        where = m.group(3)
        fields = fields.replace(" ", "")  # simple pass
        if where:
            # SUPER-basic translations
            # 'Name LIKE \'%foo%\'' and 'Field = value' → leave mostly as-is
            where_sql = (where
                .replace(" true", " 1")
                .replace(" false", " 0"))
            return f"SELECT {fields} FROM {table} WHERE {where_sql}"
        return f"SELECT {fields} FROM {table}"

    # SOSL: FIND {term} IN NAME FIELDS RETURNING Lead(Name, Phone)
    m = re.fullmatch(
        r"FIND\s+\{\s*(.+?)\s*\}\s+IN\s+NAME\s+FIELDS\s+RETURNING\s+([A-Za-z_][A-Za-z0-9_]*)\((.+?)\)",
        s, re.I
    )
    if m:
        term = m.group(1).strip().replace("'", "''")
        table = m.group(2).strip()
        fields = m.group(3).strip().replace(" ", "")
        # naive mapping: searching Name field
        return f"SELECT {fields} FROM {table} WHERE Name LIKE '%{term}%'"

    # Fallback: assume it is already SQL
    return s
