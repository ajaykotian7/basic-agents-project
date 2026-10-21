from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("MY_OPEN_AI_KEY"))

response = client.responses.create(
    model="gpt-6-astra", input="Write a short bedtime story about a unicorn."
)

print(response.output_text)
