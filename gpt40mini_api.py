# https://yourgpt.ai/tools/openai-and-other-llm-api-pricing-calculator
# https://openai.com/api/pricing/

# GPT 4o-mini is the cheapest at present, at $0.75 (0.15 input + 0.6 output) / 1M (10 Lakh)(1_000_000) tokens 
# Minimum purchase => $5 credits => should give me more than 5_000_000 tokens (including input and output)
# Track usage: https://platform.openai.com/usage

# Example:
# Input: "Tell me about yourself in one line" => 14 tokens
# Output: "I'm an AI designed to assist and provide information on a wide range of topics." => 17 tokens
# Total 31 tokens consumed in this interaction

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response_stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is a GPT?",
        },
    ],
    stream=True,
)

for chunk in response_stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

print()