import json
import os
from datetime import datetime
from typing import Optional, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# Hide the per-request HTTP logs from the OpenAI client
logging.getLogger("httpx").setLevel(logging.WARNING)


load_dotenv()

client = OpenAI(api_key=os.environ.get("MY_OPEN_AI_KEY"))


class DetectRequestType(BaseModel):
    """First LLM call: Extract request type information"""

    description: str = Field(description="Raw description of the request")
    request_type: Literal["new_event", "modify_event", "other"] = Field(
        description="Type of request whether it is a new calendar event, modification of existing event, or other type. Calendar will always contain certain required parameters : Date and participants. Otherwise is other request type"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


def extract_request_type(user_input: str):
    logger.info("Classifying request type")

    response = client.chat.completions.parse(
        model="gpt-6-astra",
        messages=[
            {
                "role": "system",
                "content": "Analyse the type of request given by the user",
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        response_format=DetectRequestType,
    )

    result = response.choices[0].message.parsed
    return result


class NewEventDetails(BaseModel):
    """Second LLM call: Parse specific event details"""

    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: list[str] = Field(description="List of participants")


def parse_new_event_info(user_input: str):
    logger.info("Parsing new event details")
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
        response_format=NewEventDetails,
    )

    result = response.choices[0].message.parsed
    return result


class Change(BaseModel):
    """Details for changing an existing event"""

    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")


class ModifyEventDetails(BaseModel):
    """Second LLM call: Parse specific event details for modification"""

    name: str = Field(description="Name of the event")
    changes: list[Change] = Field(description="List of changes to be made.")
    participants_to_add: list[str] = Field(description="List of participants to add")
    participants_to_remove: list[str] = Field(
        description="List of participants to remove"
    )


def parse_modify_event_info(user_input: str):
    logger.info("Parsing event changes")
    today = datetime.now()
    date_context = f"Today is {today.strftime('%d/%m/%Y')}"

    response = client.chat.completions.parse(
        model="gpt-6-astra",
        messages=[
            {
                "role": "system",
                "content": f"{date_context}. Parse the update existing calendar event info in the required format",
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        response_format=ModifyEventDetails,
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


def generate_event_confirmation(event_details: NewEventDetails):
    logger.info("Generating confirmation email")
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


def generate_modify_event_confirmation(event_details: ModifyEventDetails):
    logger.info("Generating update confirmation email")
    response = client.chat.completions.parse(
        model="gpt-6-astra",
        messages=[
            {
                "role": "system",
                "content": "Generate a natural confirmation email for the update in the event. Sign off with your name : Ajay Kotian",
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


def generate_failed_event_confirmation(user_input: str):
    logger.info("Generating reply for unsupported request")
    response = client.chat.completions.parse(
        model="gpt-6-astra",
        messages=[
            {
                "role": "system",
                "content": "Generate a natural confirmation email for saying that the request you shared doesnt match with any of the calendar event actions like creation/modification. Sign off with your name : Ajay Kotian",
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        response_format=EventConfirmation,
    )

    result = response.choices[0].message.parsed
    return result


def process_event_info(user_input: str):
    user_input = user_input.strip()
    print_heading(f"Request: {user_input}")

    email_response = None
    request_type = extract_request_type(user_input)
    print(
        f"Routed to: {request_type.request_type}"
        f" (confidence {request_type.confidence_score:.2f})\n"
    )
    if request_type.request_type == "new_event":
        new_event_info = parse_new_event_info(user_input)
        if new_event_info:
            print_new_event_details(new_event_info)
            email_response = generate_event_confirmation(new_event_info)
        else:
            print("Could not parse the new event details.")
    elif request_type.request_type == "modify_event":
        modify_event_info = parse_modify_event_info(user_input)
        if modify_event_info:
            print_modify_event_details(modify_event_info)
            email_response = generate_modify_event_confirmation(modify_event_info)
        else:
            print("Could not parse the event changes.")
    else:
        email_response = generate_failed_event_confirmation(user_input)
    if email_response:
        print_event_info(email_response)


def print_heading(title: str):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")


def print_new_event_details(event_details: NewEventDetails):
    print("New event:")
    print(f"  Name:         {event_details.name}")
    print(f"  Date:         {event_details.date}")
    print(f"  Duration:     {event_details.duration_minutes} minutes")
    print(f"  Participants: {', '.join(event_details.participants)}")


def print_modify_event_details(event_details: ModifyEventDetails):
    print(f"Changes to: {event_details.name}")
    for change in event_details.changes:
        print(f"  - {change.field}: {change.new_value}")
    if event_details.participants_to_add:
        print(f"  Add:    {', '.join(event_details.participants_to_add)}")
    if event_details.participants_to_remove:
        print(f"  Remove: {', '.join(event_details.participants_to_remove)}")


def print_event_info(event_confirmation: EventConfirmation):
    print("\n--- Confirmation email ---\n")
    print(event_confirmation.confirmation_message)
    if event_confirmation.calendar_link:
        print(f"\nCalendar link: {event_confirmation.calendar_link}")


response = process_event_info("""
Lets schedule a 1hr meeting next friday with Alice and bob to discuss project scope
""")

response = process_event_info("""
Update the last meeting to 1hr meeting this friday instead of last friday with Alice and bob to discuss project scope. remove Asher from the meeting as he is not needed.
""")


response = process_event_info("""
Ask Alice and bob to discuss project scope
""")

response = process_event_info("""
How's the weather in Mumbai?
""")
