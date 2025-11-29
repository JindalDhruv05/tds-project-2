from langgraph.graph import StateGraph, END, START
from shared_store import url_time
import time
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from tools import (
    get_rendered_html, download_file, post_request, get_request,
    run_code, add_dependencies, ocr_image_tool, transcribe_audio, 
    encode_image_to_base64, process_csv, interpret_instruction
)
from typing import TypedDict, Annotated, List
from langchain_core.messages import trim_messages, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
import re
import uuid
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

RECURSION_LIMIT = 5000
MAX_TOKENS = 60000


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


# Included all tools to ensure capabilities
TOOLS = [
    run_code, get_rendered_html, download_file,
    post_request, get_request, add_dependencies, 
    ocr_image_tool, transcribe_audio, encode_image_to_base64,
    process_csv, interpret_instruction
]


rate_limiter = InMemoryRateLimiter(
    requests_per_second=4 / 60,
    check_every_n_seconds=1,
    max_bucket_size=4
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter
).bind_tools(TOOLS)


SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.
Your email: {EMAIL}
Your secret: {SECRET}

GENERAL BEHAVIOUR
-----------------
- Always start by calling get_rendered_html(url) on the current quiz URL.
- Use ONLY the tools provided.
- Never invent or guess new URLs. Only use:
  * URLs you see in the HTML (links, script src, img src, etc.)
  * URLs returned by the server responses (e.g. "url" field)
- Perform all non-trivial calculations using run_code, not in your head.

AUDIO WORKFLOW
--------------
If the HTML (or instructions) include an <audio> tag or an audio URL:
  1. Use download_file(audio_url, "task.opus")
  2. Use transcribe_audio("task.opus")
  3. The transcription will contain instructions. Follow them exactly.

IMAGE / OCR WORKFLOW
--------------------
If get_rendered_html(url) returns any image URLs in the "images" list,
OR the page clearly shows a puzzle image:

  1. For each relevant image URL, call:
       download_file(image_url, "task_image.png")
  2. Then call:
       ocr_image_tool(image_path="task_image.png")
  3. Read the extracted text carefully. It may contain:
       - The puzzle statement
       - Hidden instructions
       - Formulas or constraints
  4. Follow those instructions using the other tools.

You MUST NOT ignore images. Any time an image is downloaded, it should be OCR'd.

ANSWER SUBMISSION
-----------------
To submit an answer, you must call post_request with a payload that includes:
  - "answer": the computed answer (string or number)
  - "email": {EMAIL}
  - "url": the quiz page URL you are answering for
"""


def handle_malformed_node(state: AgentState):
    print("--- DETECTED MALFORMED JSON. ASKING AGENT TO RETRY ---")
    return {
        "messages": [
            HumanMessage(
                content=(
                    "SYSTEM ERROR: Your last tool call had invalid JSON or malformed arguments. "
                    "Please call the same tool again, but this time emit ONLY valid JSON."
                )
            )
        ]
    }


def agent_node(state: AgentState):
    # --- TIME HANDLING ---
    cur_time = time.time()
    cur_url = os.getenv("url")
    prev_time = url_time.get(cur_url) 
    offset = os.getenv("offset", "0")

    if prev_time is not None:
        prev_time = float(prev_time)
        diff = cur_time - prev_time
        if diff >= 180 or (offset != "0" and (cur_time - float(offset)) > 90):
            print(f"Timeout exceeded ({diff}s) — submitting wrong answer to skip.")
            fail_msg = HumanMessage(content="You have exceeded the time limit. Submit a WRONG answer immediately to proceed.")
            result = llm.invoke(state["messages"] + [fail_msg])
            return {"messages": [result]}
    # ---------------------

    trimmed_messages = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm, 
    )
    
    if not any(msg.type == "human" for msg in trimmed_messages):
        current_url = os.getenv("url", "Unknown URL")
        trimmed_messages.append(HumanMessage(content=f"Context cleared. Continue processing URL: {current_url}"))

    # --- FORCED TOOL EXECUTION LOGIC ---
    recent_messages = trimmed_messages[-15:]
    msg_str = str(recent_messages)

    # 1. Detect Audio
    downloaded_audio = None
    transcribed_audio = "transcribe_audio" in msg_str
    
    # 2. Detect Image
    downloaded_image = None
    ocr_run = "ocr_image_tool" in msg_str

    for msg in recent_messages:
        content = str(msg.content)
        # Check Audio (optimized regex)
        audio_match = re.search(r'Downloaded file to:.*?([\w-]+\.(opus|mp3|wav|ogg))', content)
        if audio_match:
            downloaded_audio = audio_match.group(1).strip()
        
        # Check Image (optimized regex)
        img_match = re.search(r'Downloaded file to:.*?([\w-]+\.(png|jpg|jpeg|bmp))', content)
        if img_match:
            downloaded_image = img_match.group(1).strip()

    # FORCE TRANSCRIPTION
    if downloaded_audio and not transcribed_audio:
        print(f"DETECTED: Audio '{downloaded_audio}' pending transcription. FORCING TOOL.")
        return {"messages": [AIMessage(content="", tool_calls=[{
            "name": "transcribe_audio",
            "args": {"file_path": downloaded_audio},
            "id": f"force_audio_{uuid.uuid4().hex[:8]}"
        }])]}

    # FORCE OCR (Simple Argument)
    if downloaded_image and not ocr_run:
        # Clean path to just filename if needed
        if "LLMFiles" in downloaded_image:
            downloaded_image = os.path.basename(downloaded_image)
            
        print(f"DETECTED: Image '{downloaded_image}' pending OCR. FORCING TOOL.")
        return {"messages": [AIMessage(content="", tool_calls=[{
            "name": "ocr_image_tool",
            "args": {"image_path": downloaded_image},
            "id": f"force_ocr_{uuid.uuid4().hex[:8]}"
        }])]}

    print(f"--- INVOKING AGENT (Context: {len(trimmed_messages)} items) ---")
    result = llm.invoke(trimmed_messages)
    return {"messages": [result]}


def route(state):
    last = state["messages"][-1]
    
    if "finish_reason" in last.response_metadata:
        if last.response_metadata["finish_reason"] == "MALFORMED_FUNCTION_CALL":
            return "handle_malformed"

    if getattr(last, "tool_calls", None):
        print("Route → tools")
        return "tools"

    content = getattr(last, "content", "")
    if isinstance(content, str) and content.strip() == "END":
        return END

    print("Route → agent")
    return "agent"


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("handle_malformed", handle_malformed_node)

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("handle_malformed", "agent")

graph.add_conditional_edges("agent", route, {
    "tools": "tools",
    "agent": "agent",
    "handle_malformed": "handle_malformed",
    END: END
})

app = graph.compile()


def run_agent(url: str):
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url}
    ]
    app.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": RECURSION_LIMIT}
    )
    print("Tasks completed successfully!")