
import os
import csv
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Project paths
ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

INPUT_FILE = ROOT / "examples.csv"
OUTPUT_FILE = ROOT / "experiments" / "baseline_results.csv"

# Initialise Groq
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL")

if not api_key or not model_name:
    raise ValueError("Groq API key or model is missing")

client = Groq(api_key=api_key)

# Read dataset
with open(INPUT_FILE, "r", encoding="utf-8-sig") as file:
    examples = list(csv.DictReader(file))

results = []

# Test every question
for example in examples:

    question = example["question"]
    expected = example["expected_answer"]

    print(f"\nTesting: {example['example_id']}")
    print(f"Question: {question}")

    start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0
        )

        latency = round(time.perf_counter() - start, 3)

        actual = response.choices[0].message.content
        usage = response.usage

        results.append({
            "example_id": example["example_id"],
            "model": model_name,
            "question": question,
            "expected_answer": expected,
            "actual_answer": actual,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "latency_seconds": latency,
            "api_status": "success",
            "correctness": "pending",
            "citation": "pending",
            "notes": ""
        })

        print(f"Answer: {actual}")
        print(f"Tokens: {usage.total_tokens}")
        print(f"Latency: {latency} seconds")

    except Exception as error:

        results.append({
            "example_id": example["example_id"],
            "model": model_name,
            "question": question,
            "expected_answer": expected,
            "actual_answer": "",
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
            "latency_seconds": round(
                time.perf_counter() - start, 3
            ),
            "api_status": "failed",
            "correctness": "not_evaluated",
            "citation": "not_evaluated",
            "notes": type(error).__name__
        })

        print(f"API error: {type(error).__name__}")

    time.sleep(2)

# Save results
OUTPUT_FILE.parent.mkdir(
    parents=True, exist_ok=True
)

if results:
    with open(
        OUTPUT_FILE, "w",
        newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nEvaluation completed.")
    print(f"Results saved to: {OUTPUT_FILE}")

else:
    print("No examples found.")