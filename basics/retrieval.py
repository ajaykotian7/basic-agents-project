import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

client = OpenAI(api_key=os.environ.get("MY_OPEN_AI_KEY"))

# Resolve relative to this file so the script works from any working directory
KB_PATH = os.path.join(os.path.dirname(__file__), "kb.json")


def search_kb(question: str):
    """Load entire Knowledge base from file"""
    with open(KB_PATH, "r") as f:
        return json.load(f)


# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "name": "search_kb",
        "description": "Get the answer to user's questions from the knowledge base",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

system_prompt = "You are a helpful that answers questions from the knowledge base about our e-commerce store"


class KBResponse(BaseModel):
    # extra="forbid" generates additionalProperties: False required for strict schemas
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(description="This is the answer to the user's question")
    source: int = Field(description="Id of the answer")


def invoke_ai(input: str):
    input_list = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": input,
        },
    ]

    response = client.responses.create(
        model="gpt-6-astra",
        input=input_list,
        tools=tools,
    )

    input_list.extend(response.output)

    def call_function(tool_call):
        if tool_call.name == "search_kb":
            return search_kb(json.loads(tool_call.arguments)["question"])

    for tool_call in response.output:
        if tool_call.type == "function_call":
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(call_function(tool_call)),
                }
            )

    # print("Final input:")
    # print(input_list)

    response = client.responses.create(
        model="gpt-6-astra",
        # previous_response_id=response.id,
        tools=tools,
        input=input_list,
        text={
            "format": {
                "type": "json_schema",
                "name": "kb_response",
                "strict": True,
                "schema": KBResponse.model_json_schema(),
            }
        },
    )

    # 5. The model should be able to give a response!
    report = KBResponse.model_validate_json(response.output_text)
    return report


def print_heading(title: str):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")


def trigger_and_print(input):
    print_heading(f"Question: {input}")
    report = invoke_ai(input)
    print(f"Answer: {report.answer}")
    print(f"Source: knowledge base record #{report.source}")


trigger_and_print("Can you ship it to me in Thane?")
trigger_and_print("Do you accept cash in Thane?")
trigger_and_print("What payment modes do you accept in Nashik?")
trigger_and_print("What payment modes do you accept in London?")
trigger_and_print("What payment modes do you accept in Kathmandu?")
trigger_and_print("How's the weather in Mumbai?")
