from google import genai
import subprocess
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from google.genai import types
load_dotenv()
client = genai.Client()

def strip_code_fences(code: str) -> str:
    code = code.strip()
    # Remove ```python ... ``` or ``` ... ```
    if code.startswith("```"):
        # remove first line (```python or ```)
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("\n", 1)[0]
    return code.strip()

@tool
def run_code(code: str) -> str:
    """
    Execute Python code and return the output.
    
    IMPORTANT: This tool returns the program output as a STRING, not a dict.
    If there's an error, it returns the error message.
    
    Parameters
    ----------
    code : str
        Python source code to execute. Can include imports and multiple statements.
        
    Returns
    -------
    str
        The stdout output from the program, or error message if execution failed.
        
    Example
    -------
    run_code("import pandas as pd\\ndf = pd.read_csv('data.csv')\\nprint(df.sum())")
    → Returns: "12345" (the sum)
    """
    try: 
        filename = "runner.py"
        os.makedirs("LLMFiles", exist_ok=True)
        with open(os.path.join("LLMFiles", filename), "w", encoding="utf-8") as f:
            f.write(code)

        proc = subprocess.Popen(
            ["uv", "run", filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="LLMFiles"
        )
        stdout, stderr = proc.communicate()
        
        # If there's an error, return it
        if proc.returncode != 0 or stderr:
            error_msg = stderr if stderr else "Unknown error"
            return f"Error executing code: {error_msg[:500]}"
        
        # Return stdout, truncated if too long
        if len(stdout) >= 10000:
            return stdout[:10000] + "...truncated"
        
        return stdout.strip()
        
    except Exception as e:
        return f"Exception: {str(e)}"