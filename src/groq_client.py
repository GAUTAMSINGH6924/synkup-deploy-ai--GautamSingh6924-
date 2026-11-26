
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing")

client = Groq(api_key=api_key)

question = (
    "When do MCA third-semester theory examinations "
    "begin at Panjab University in December 2026?"
)

response = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL"),
    messages=[
        {"role": "user", "content": question}
    ],
    temperature=0
)

print("\nQuestion:")
print(question)

print("\nAI Response:")
print(response.choices[0].message.content)

print("\nToken Usage:")
print(response.usage)
