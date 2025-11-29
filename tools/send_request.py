from langchain_core.tools import tool
from shared_store import BASE64_STORE, url_time
import time
import os
import requests
import json
from collections import defaultdict
from typing import Any, Dict, Optional

cache = defaultdict(int)
retry_limit = 4
SECRET = os.getenv("SECRET")



@tool
def get_request(url: str, headers: Optional[Dict[str, str]] = None) -> Any:
    """
    Send an HTTP GET request to the given URL.
    
    Use this for fetching data from APIs, especially paginated endpoints like /api/items?page=1.
    This is the correct tool for retrieving data, not posting data.
    REMEMBER: This is a blocking function so it may take a while to return. Wait for the response.
    
    Args:
        url (str): The endpoint to send the GET request to.
        headers (Optional[Dict[str, str]]): Optional HTTP headers to include in the request.
    
    Returns:
        Any: The response body. If the server returns JSON, a parsed dict is returned.
        Otherwise, the raw text response is returned.
    
    Raises:
        requests.HTTPError: If the server responds with an unsuccessful status.
        requests.RequestException: For network-related errors.
    """
    from urllib.parse import urlparse
    
    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return {"error": f"Invalid URL: {url}. Must be a complete URL with https://"}
    
    # Prevent hallucinated URLs
    blocked_domains = ["example.com", "YOUR_", "quiz.example.com", "api.example.com"]
    if any(blocked in url for blocked in blocked_domains):
        return {"error": f"Refusing to send to placeholder URL: {url}. Use the actual URL from the page."}
    
    headers = headers or {"Content-Type": "application/json"}
    
    try:
        print(f"\nFetching from URL: {url}\n")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        try:
            data = response.json()
            print("Got JSON response:\n", json.dumps(data, indent=4), '\n')
            return data
        except ValueError:
            # Return full text, but print preview
            full_text = response.text
            preview = full_text[:500]
            if len(full_text) > 500:
                print(f"Got text response ({len(full_text)} chars, showing first 500):\n{preview}\n...")
            else:
                print(f"Got text response:\n{preview}\n")
            return full_text
            
    except requests.HTTPError as e:
        err_resp = e.response
        try:
            err_data = err_resp.json()
        except ValueError:
            err_data = err_resp.text
        print("HTTP Error Response:\n", err_data)
        return {"error": err_data, "status_code": err_resp.status_code}
        
    except Exception as e:
        print("Unexpected error:", e)
        return {"error": str(e)}

@tool
def post_request(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
    """
    Send an HTTP POST request to the given URL with the provided payload.

    This function is designed for LangGraph applications, where it can be wrapped
    as a Tool or used inside a Runnable to call external APIs, webhooks, or backend
    services during graph execution.
    REMEMBER: This a blocking function so it may take a while to return. Wait for the response.

    Args:
        url (str): The endpoint to send the POST request to.
        payload (Dict[str, Any]): The JSON-serializable request body.
        headers (Optional[Dict[str, str]]): Optional HTTP headers to include
            in the request. If omitted, a default JSON header is applied.

    Returns:
        Any: The response body. If the server returns JSON, a parsed dict is
        returned. Otherwise, the raw text response is returned.

    Raises:
        requests.HTTPError: If the server responds with an unsuccessful status.
        requests.RequestException: For network-related errors.
    """
    # Handling if the answer is a BASE64
    ans = payload.get("answer")

    if isinstance(ans, str) and ans.startswith("BASE64_KEY:"):
        key = ans.split(":", 1)[1]
        payload["answer"] = BASE64_STORE[key]

    # 🔐 ALWAYS inject secret so we never get "Missing field secret"
    global SECRET
    if SECRET is None:
        SECRET = os.getenv("SECRET")
    if SECRET and not payload.get("secret"):
        payload["secret"] = SECRET
    
    # Validate URL before sending
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return {"error": f"Invalid URL: {url}. Must be a complete URL with https://"}
    
    # Prevent hallucinated URLs
    blocked_domains = ["example.com", "YOUR_", "quiz.example.com", "api.example.com"]
    if any(blocked in url for blocked in blocked_domains):
        return {"error": f"Refusing to send to placeholder URL: {url}. Use the actual URL from the page."}
    
    headers = headers or {"Content-Type": "application/json"}

    try:
        cur_url = os.getenv("url")
        cache[cur_url] += 1

        # For logging, avoid printing giant payloads
        sending = {
            "answer": str(payload.get("answer", ""))[:100],
            "email": payload.get("email", ""),
            "url": payload.get("url", ""),
            # don't print full secret, just indicate presence
            "has_secret": bool(payload.get("secret")),
        }
        print(f"\nSending Answer \n{json.dumps(sending, indent=4)}\n to url: {url}")
        response = requests.post(url, json=payload, headers=headers)

        # Raise on 4xx/5xx
        response.raise_for_status()

        # Try to return JSON, fallback to raw text
        try:
            data = response.json()
        except ValueError:
            print("Non-JSON response from server")
            return {"error": "non_json_response", "text": response.text}

        print("Got the response: \n", json.dumps(data, indent=4), '\n')
        
        delay = time.time() - url_time.get(cur_url, time.time())
        print(delay)

        next_url = data.get("url")

        # === TERMINATION CONDITION: final quiz, no next URL ===
        if not next_url:
            print("No next URL in server response. Signaling END to the agent.")
            return {
                "agent_signal": "END",
                "response": data
            }

        # Track next_url timing
        if next_url not in url_time:
            url_time[next_url] = time.time()

        correct = data.get("correct")
        if not correct:
            cur_time = time.time()
            prev = url_time.get(next_url, time.time())
            # Shouldn't retry
            if cache[cur_url] >= retry_limit or delay >= 180 or (prev != "0" and (cur_time - float(prev)) > 90):
                print("Not retrying, moving on to the next question")
                data = {"url": data.get("url", "")}
            else:
                os.environ["offset"] = str(url_time.get(next_url, time.time()))
                print("Retrying..")
                data["url"] = cur_url
                data["message"] = "Retry Again!"

        print("Formatted: \n", json.dumps(data, indent=4), '\n')
        forward_url = data.get("url", "")
        os.environ["url"] = forward_url 
        if forward_url == next_url:
            os.environ["offset"] = "0"

        return data

    except requests.HTTPError as e:
        # Extract server’s error response
        err_resp = e.response

        try:
            err_data = err_resp.json()
        except ValueError:
            err_data = err_resp.text

        print("HTTP Error Response:\n", err_data)
        return {"error": err_data, "status_code": err_resp.status_code}

    except Exception as e:
        print("Unexpected error:", e)
        return {"error": str(e)}
