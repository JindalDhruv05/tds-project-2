import csv
from io import StringIO
from typing import Dict, Any, List
from langchain_core.tools import tool


def _apply_filters(row_values: List[float], filters: List[Dict[str, Any]]) -> bool:
    """
    Return True if the row passes ALL filters.
    Each filter is of the form:
      {"column": int, "op": str, "value": number}
    """
    for f in filters:
        try:
            col = int(f.get("column", 0))
            op = f.get("op", "==")
            val = float(f.get("value", 0))
            x = float(row_values[col])
        except Exception:
            return False

        if op == ">":
            if not (x > val):
                return False
        elif op == ">=":
            if not (x >= val):
                return False
        elif op == "<":
            if not (x < val):
                return False
        elif op == "<=":
            if not (x <= val):
                return False
        elif op == "==":
            if not (x == val):
                return False
        elif op == "!=":
            if not (x != val):
                return False
        else:
            return False

    return True


@tool
def process_csv(csv_text: str, operations: Dict[str, Any]) -> Dict[str, Any]:
    """
    General CSV processor that applies the structured 'operations' object produced
    by interpret_instruction.

    operations schema (typical):
        {
          "operation": "sum" | "count" | "max" | "min" | "average",
          "column": 0,
          "filters": [
            {"column": 0, "op": ">=", "value": 50000}
          ]
        }

    Returns a dict with at least:
        {"result": <numeric>, "rows_matched": int}
    """
    op_name = str(operations.get("operation", "sum")).lower()
    col_idx = int(operations.get("column", 0))
    filters = operations.get("filters", [])
    if not isinstance(filters, list):
        filters = []

    f = StringIO(csv_text)
    reader = csv.reader(f)

    values = []
    rows_matched = 0

    for row in reader:
        if not row:
            continue

        try:
            # Pad row if shorter than column index
            if col_idx >= len(row):
                continue
            row_vals = [float(x) for x in row]
        except Exception:
            continue

        if not _apply_filters(row_vals, filters):
            continue

        try:
            v = float(row_vals[col_idx])
        except Exception:
            continue

        values.append(v)
        rows_matched += 1

    if not values:
        result = 0
    else:
        if op_name in ["sum", "total"]:
            result = sum(values)
        elif op_name == "count":
            result = len(values)
        elif op_name == "max":
            result = max(values)
        elif op_name == "min":
            result = min(values)
        elif op_name in ["avg", "average", "mean"]:
            result = sum(values) / len(values)
        else:  # unknown op → default to sum
            result = sum(values)

    return {
        "result": result,
        "rows_matched": rows_matched,
        "operation": op_name,
        "column": col_idx,
        "filters": filters,
    }
    