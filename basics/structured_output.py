from openai import OpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

client = OpenAI(api_key=os.environ.get("MY_OPEN_AI_KEY"))


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


class CalendarEventList(BaseModel):
    events: list[CalendarEvent]


def print_heading(title: str):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")


user_input = "Alice and Bob are going to the fair on Friday. Ajay and Arjun are going to the Doctor on Tuesday"

response = client.chat.completions.parse(
    model="gpt-6-astra",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": user_input,
        },
    ],
    response_format=CalendarEventList,
)

eventList = response.choices[0].message.parsed

print_heading(f"Input: {user_input}")
print(f"Extracted {len(eventList.events)} event(s):\n")
for i, event in enumerate(eventList.events, start=1):
    print(f"{i}. {event.name}")
    print(f"   Date:         {event.date}")
    print(f"   Participants: {', '.join(event.participants)}\n")
