import json
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model


INSTRUCTION_SYSTEM_PROMPT = """
You are a helper model that converts NATURAL LANGUAGE instructions about CSV processing
into a STRICT JSON object that another program can execute.

You MUST respond with ONLY a JSON object, nothing else, matching this schema:

{
  "operation": "sum" | "count" | "max" | "min" | "average",
  "column": <integer index of the column to operate on, 0-based>,
  "filters": [
    {
      "column": <integer column index, 0-based>,
      "op": ">" | ">=" | "<" | "<=" | "==" | "!=",
      "value": <number>
    }
  ]
}

Rules:
- Assume numeric data.
- If the user says "first column", use column index 0.
- If they mention a cutoff like "greater than or equal to 50000", that becomes a filter
  with op ">=" and value 50000.
- If they say "greater than cutoff value provided", assume the caller knows the numeric cutoff
  and will adjust the filter accordingly.
- If no filter is described, use an empty list for "filters".
- If they say "add", "sum", "total", or "sum up", use operation "sum".
- If they say "count how many", use "count".
- If they say "largest", use "max".
- If they say "smallest", use "min".
- If they say "average" or "mean", use "average".

Output ONLY the JSON, no explanation.
"""


@tool
def interpret_instruction(text: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Use a small LLM to convert natural-language instructions about CSV processing
    into a structured JSON object describing the operation and filters.

    Args:
        text: The transcribed instruction (e.g., from audio).
        context: Optional extra context (HTML snippet, URL, notes).

    Returns:
        A dict describing the operation, or an error payload if parsing failed.
    """
    llm = init_chat_model(
        model_provider="google_genai",
        model="gemini-2.5-flash",
    )

    user_content = f"Instruction:\n{text}\n\nContext:\n{context or ''}"

    msg = [
        {"role": "system", "content": INSTRUCTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw = llm.invoke(msg)
    content = getattr(raw, "content", "") if raw is not None else ""

    # Sometimes models wrap JSON in ```json ... ```; strip that if present.
    content_str = str(content).strip()
    if content_str.startswith("```"):
        content_str = content_str.strip("`")
        # remove possible leading "json"
        if content_str.lower().startswith("json"):
            content_str = content_str[4:].lstrip()

    try:
        data = json.loads(content_str)
        # basic validation
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON is not an object")
        return data
    except Exception as e:
        return {
            "error": "could_not_parse_json",
            "reason": str(e),
            "raw": content_str,
        }
