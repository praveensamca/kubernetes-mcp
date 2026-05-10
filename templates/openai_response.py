import os
from openai import OpenAI
import json

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=API_KEY)

response = client.responses.create(
    model="gpt-5.4-mini",
    input="what is the day today? and time now , where am i?",
    store=True,
)

print(json.loads(response.model_dump_json(indent=2))["output"][0]["content"][0]["text"])
