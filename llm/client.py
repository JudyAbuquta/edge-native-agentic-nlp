import os
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv


# Load environment variables from .env (if present)
load_dotenv()

# Read API key from environment
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Please add it to a .env file or your environment variables."
    )

# Configure Gemini client
genai.configure(api_key=API_KEY)

# Choose a default model (you can change this later if needed)
DEFAULT_MODEL = "gemini-2.5-flash"


def call_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """
    Send a prompt to Gemini and return the text response.
    - prompt: user prompt or message
    - system_instruction: optional 'role' / behavior description for the model
    - model_name: which Gemini model to use
    """
    try:
        if system_instruction:
            model = genai.GenerativeModel(
                model_name,
                system_instruction=system_instruction
            )
        else:
            model = genai.GenerativeModel(model_name)

        response = model.generate_content(prompt)

        # Safely get the response text
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return "[LLM returned no text response]"

    except Exception as e:
        # You can log this if you want
        return f"[LLM ERROR] {e}"
