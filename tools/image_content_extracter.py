import base64
import os
from langchain_core.tools import tool
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Initialize Client (reuses your existing GOOGLE_API_KEY)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@tool
def ocr_image_tool(image_path: str) -> str:
    """
    Extract text and analyze content from an image file using the Gemini Vision model.
    
    Args:
        image_path (str): The filename of the image (e.g., "sample_image.png"). 
                          The tool automatically looks in the 'LLMFiles' directory.

    Returns:
        str: The extracted text or description of the image content.
    """
    try:
        # 1. Resolve full path
        if not image_path.startswith("LLMFiles"):
            full_path = os.path.join("LLMFiles", image_path)
        else:
            full_path = image_path

        if not os.path.exists(full_path):
            return f"Error: Image file not found at {full_path}"

        print(f"Processing image with Gemini Vision: {full_path}")

        # 2. Read Image Bytes
        with open(full_path, "rb") as f:
            image_bytes = f.read()

        # 3. Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/png"  # generic mime type usually works
                        ),
                        types.Part.from_text(
                            text="Transcribe ALL text, numbers, and codes visible in this image exactly as they appear. "
                                 "If it is a puzzle, describe the visual elements needed to solve it."
                        )
                    ]
                )
            ]
        )
        
        return response.text.strip()

    except Exception as e:
        return f"Error during AI Image Analysis: {str(e)}"