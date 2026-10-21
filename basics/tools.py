from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
import requests
import json

load_dotenv()

client = OpenAI(api_key=os.environ.get("MY_OPEN_AI_KEY"))


def get_weather(latitude, longitude):
    """This is publically available api"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    response = requests.get(url=url)
    data = response.json()
    return data["current"]


# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

system_prompt = "You are a helpful weather assistant"

question = "What's the weather like in Thane, Maharasthra, India today"

input_list = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": question,
    },
]


def print_heading(title: str):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")


print_heading(f"Question: {question}")

response = client.responses.create(
    model="gpt-6-astra",
    input=input_list,
    tools=tools,
)

input_list.extend(response.output)

for item in response.output:
    if item.type == "function_call":
        if item.name == "get_weather":
            # 3. Execute the function logic for get_weather
            arguments = json.loads(item.arguments)
            latitude = arguments["latitude"]
            longitude = arguments["longitude"]
            print(
                f"Tool call:   get_weather(latitude={latitude}, longitude={longitude})"
            )
            weather = get_weather(latitude, longitude)
            print(f"Tool result: {json.dumps(weather)}")

            # 4. Provide function call results to the model
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(weather),
                }
            )


class WeatherReport(BaseModel):
    # extra="forbid" generates additionalProperties: False required for strict schemas
    model_config = ConfigDict(extra="forbid")

    location: str = Field(description="Name of the city or region")
    temperature_celsius: float = Field(description="Current temperature in Celsius")
    summary: str = Field(
        description="A concise 1-2 sentence description of the conditions"
    )
    clothing_advice: str = Field(description="Practical clothing recommendation")


response = client.responses.create(
    model="gpt-6-astra",
    # previous_response_id=response.id,
    tools=tools,
    input=input_list,
    text={
        "format": {
            "type": "json_schema",
            "name": "weather_report",
            "strict": True,
            "schema": WeatherReport.model_json_schema(),
        }
    },
)

# 5. The model should be able to give a response!
report = WeatherReport.model_validate_json(response.output_text)

print("\n--- Weather Report ---\n")
print(f"Location:    {report.location}")
print(f"Temperature: {report.temperature_celsius}°C")
print(f"Summary:     {report.summary}")
print(f"Advice:      {report.clothing_advice}")
