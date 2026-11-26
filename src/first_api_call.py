
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Read the API key and model name
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

# Initialise the Gemini client
client = genai.Client(api_key=api_key)

# Student question
question = (
    "When do MCA third-semester theory examinations "
    "begin at Panjab University in December 2026?"
)

# Send the question to the AI model
response = client.models.generate_content(
    model=model_name,
    contents=question
)

# Display the result
print("\nQuestion:")
print(question)

print("\nAI Response:")
print(response.text)

print("\nToken Usage:")
print(response.usage_metadata)