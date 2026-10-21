import json
import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

client = OpenAI(api_key=os.environ.get("MY_OPEN_AI_KEY"))


class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information"""

    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


def extract_event_info(user_input: str):
    today = datetime.now()
    date_context = f"Today is {today.strftime('%d/%m/%Y')}"

    response = client.chat.completions.parse(
        model="gpt-6-astra",
        messages=[
            {
                "role": "system",
                "content": f"{date_context}. Analyse if the input is calendar event information",
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        response_format=EventExtraction,
    )

    result = response.choices[0].message.parsed
    return result


class EventDetails(BaseModel):
    """Second LLM call: Parse specific event details"""

    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: list[str] = Field(description="List of participants")


def parse_event_info(user_input: str):
    today = datetime.now()
    date_context = f"Today is {today.strftime('%d/%m/%Y')}"

    response = client.chat.completions.parse(
        model="gpt-6-astra",
        messages=[
            {
                "role": "system",
                "content": f"{date_context}. Parse the calendar info in the required format",
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        response_format=EventDetails,
    )

    result = response.choices[0].message.parsed
    return result


class EventConfirmation(BaseModel):
    """Third LLM call: Generate confirmation message"""

    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated google calendar link if applicable which on opening in browser will take to the calendar page"
    )


def generate_event_confirmation(event_details: EventDetails):
    response = client.chat.completions.parse(
        model="gpt-6-astra",
        messages=[
            {
                "role": "system",
                "content": "Generate a natural confirmation email for the event. Sign off with your name : Ajay Kotian",
            },
            {
                "role": "user",
                "content": str(event_details.model_dump()),
            },
        ],
        response_format=EventConfirmation,
    )

    result = response.choices[0].message.parsed
    return result


def process_event_info(user_input: str):
    user_input = user_input.strip()
    print_heading(f"Request: {user_input}")

    print("Step 1: Check if this is a calendar event")
    initial_extraction = extract_event_info(user_input)
    print(
        f"  Calendar event: {'Yes' if initial_extraction.is_calendar_event else 'No'}"
    )
    print(f"  Confidence:     {initial_extraction.confidence_score:.2f}")
    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confidence_score < 0.8
    ):
        print("\nStopped: needs a calendar event with confidence of at least 0.80.")
        return None

    print("\nStep 2: Parse event details")
    event_details = parse_event_info(initial_extraction.description)
    print_event_details(event_details)

    print("\nStep 3: Generate confirmation email")
    event_confirmation = generate_event_confirmation(event_details)
    print_event_info(event_confirmation)
    return event_confirmation


def print_heading(title: str):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")


def print_event_details(event_details: EventDetails):
    print(f"  Name:         {event_details.name}")
    print(f"  Date:         {event_details.date}")
    print(f"  Duration:     {event_details.duration_minutes} minutes")
    print(f"  Participants: {', '.join(event_details.participants)}")


def print_event_info(event_confirmation: EventConfirmation):
    print("\n--- Confirmation email ---\n")
    print(event_confirmation.confirmation_message)
    if event_confirmation.calendar_link:
        print(f"\nCalendar link: {event_confirmation.calendar_link}")


response = process_event_info("""
Lets schedule a 1hr meeting next friday with Alice and bob to discuss project scope
""")


response = process_event_info("""
Ask Alice and bob to discuss project scope
""")
