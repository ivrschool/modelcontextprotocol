import os
from dotenv import load_dotenv
from google import genai


def configure_gemini():
    """
    Load the Gemini API key from environment variables and initialize the Gemini client.

    Returns:
        genai.Client: A configured Gemini client instance.

    Raises:
        ValueError: If the API key is not found in the environment.
    """
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")

    return genai.Client(api_key=gemini_api_key)
