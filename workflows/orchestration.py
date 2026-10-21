import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

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


class SectionPlan(BaseModel):
    "Blog Section Plan Details"

    section_type: str = Field(description="Type of section to write")
    section_description: str = Field(description="Description of section to write")
    style_guide: str = Field(
        description="Writing Style to consider while writing the section"
    )
    target_length: int = Field(
        description="Number of words that the section can roughly have"
    )


class OrchestratorPlan(BaseModel):
    "Blog Plan Details"

    title: str = Field(description="Apt Eyecatching Title for the Blog")
    topic_analysis: str = Field(description="Brief description of the topic")
    audience: str = Field(description="Description of intended audience")
    sections: list[SectionPlan] = Field(description="List of sections to be written")


class SectionContent(BaseModel):
    """Content written by a worker"""

    content: str = Field(description="Written content for the section")
    key_points: list[str] = Field(description="Main points covered")


class SuggestedEdits(BaseModel):
    """Suggested edits for a section"""

    section_name: str = Field(description="Name of the section")
    suggested_edit: str = Field(description="Suggested edit")


class ReviewFeedback(BaseModel):
    """Final review and suggestions"""

    cohesion_score: float = Field(description="How well sections flow together (0-1)")
    suggested_edits: list[SuggestedEdits] = Field(
        description="Suggested edits by section"
    )
    final_version: str = Field(description="Complete, polished blog post")


ORCHESTRATOR_PROMPT = """
Analyze this blog topic and break it down into logical sections.

Topic: {topic}
Target Length: {target_length} words
Style: {style}

Return your response in this format:

# Analysis
Analyze the topic and explain how it should be structured.
Consider the narrative flow and how sections will work together.

# Target Audience
Define the target audience and their interests/needs.

# Sections
## Section 1
- Type: section_type
- Description: what this section should cover
- Style: writing style guidelines

[Additional sections as needed...]
"""

WORKER_PROMPT = """
Write a blog section based on:
Topic: {topic}
Section Type: {section_type}
Section Goal: {description}
Style Guide: {style_guide}
Target Length : {target_length} words
Previous Sections : {previous_sections}

Put only the section prose in `content`, following the style guide. Do not add a "Content" heading.
List the main points covered in `key_points`.
"""

REVIEWER_PROMPT = """
Review this blog post for cohesion and flow:

Topic: {topic}
Target Audience: {audience}

Sections:
{sections}

Provide a cohesion score between 0.0 and 1.0, suggested edits for each section if needed, and a final polished version of the complete post.

The cohesion score should reflect how well the sections flow together, with 1.0 being perfect cohesion.
For suggested edits, focus on improving transitions and maintaining consistent tone across sections.
The final version should incorporate your suggested improvements into a polished, cohesive blog post.
"""


class BlogOrchestrator:
    def __init__(self):
        self.sections_content: list[SectionContent] = []
        self.title: str

    def get_plan(self, topic: str, style: str, target_length: int) -> OrchestratorPlan:
        response = client.chat.completions.parse(
            model="gpt-6-astra",
            messages=[
                {
                    "role": "system",
                    "content": ORCHESTRATOR_PROMPT.format(
                        topic=topic, target_length=target_length, style=style
                    ),
                },
            ],
            response_format=OrchestratorPlan,
        )

        result = response.choices[0].message.parsed
        self.title = result.title
        return result

    def write_blog(self, topic: str, sections: list[SectionPlan]):
        for i, section in enumerate(sections, start=1):
            logger.info(f"Writing section {i}/{len(sections)}: {section.section_type}")
            self.write_and_append_section(topic, section)
        return self.sections_content

    def format_sections(self) -> str:
        """Render written sections as plain text for use inside a prompt"""
        return "\n\n".join(
            f"## Section {i}\n{section.content}"
            for i, section in enumerate(self.sections_content, start=1)
        )

    def write_and_append_section(self, topic: str, section: SectionPlan):
        previous_sections = self.format_sections()

        response = client.chat.completions.parse(
            model="gpt-6-astra",
            messages=[
                {
                    "role": "system",
                    "content": WORKER_PROMPT.format(
                        topic=topic,
                        section_type=section.section_type,
                        description=section.section_description,
                        style_guide=section.style_guide,
                        target_length=section.target_length,
                        previous_sections=previous_sections
                        if previous_sections
                        else "This is the first section.",
                    ),
                },
            ],
            response_format=SectionContent,
        )
        result = response.choices[0].message.parsed
        self.sections_content.append(result)

    def review_post(self, topic: str, plan: OrchestratorPlan) -> ReviewFeedback:
        """Reviewer: Analyze and improve overall cohesion"""

        completion = client.chat.completions.parse(
            model="gpt-6-astra",
            messages=[
                {
                    "role": "system",
                    "content": REVIEWER_PROMPT.format(
                        topic=topic,
                        audience=plan.audience,
                        sections=self.format_sections(),
                    ),
                }
            ],
            response_format=ReviewFeedback,
        )
        return completion.choices[0].message.parsed

    def print_plan(self, plan: OrchestratorPlan):
        print_heading("BLOG PLAN")
        print(f"Title:    {plan.title}")
        print(f"Audience: {plan.audience}\n")
        print(f"Analysis:\n{clean(plan.topic_analysis)}\n")
        print("Sections:")
        for i, section in enumerate(plan.sections, start=1):
            print(f"  {i}. {section.section_type} (~{section.target_length} words)")
            print(f"     {section.section_description}")

    def print_blog(self, plan: OrchestratorPlan):
        print_heading(f"DRAFT: {self.title}")
        for i, (section_plan, section) in enumerate(
            zip(plan.sections, self.sections_content), start=1
        ):
            print_subheading(f"Section {i}: {section_plan.section_type}")
            print(clean(section.content))
            print("\nKey points:")
            for point in section.key_points:
                print(f"  - {point}")
        return self.sections_content

    def print_review(self, review: ReviewFeedback):
        print_heading("REVIEW")
        print(f"Cohesion score: {review.cohesion_score:.2f} / 1.00\n")
        print("Suggested edits:")
        if not review.suggested_edits:
            print("  (none)")
        for edit in review.suggested_edits:
            print(f"  - {edit.section_name}: {edit.suggested_edit}")

        print_heading(f"FINAL VERSION: {self.title}")
        print(clean(review.final_version))


def clean(text: str) -> str:
    """Turn escaped newlines from the model into real line breaks"""
    return text.replace("\\n", "\n").strip()


def print_heading(title: str):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")


def print_subheading(title: str):
    print(f"\n--- {title} ---\n")


bo = BlogOrchestrator()
logger.info("Planning the blog")
plan = bo.get_plan(
    topic="Pokemon canto region",
    style="Dry academic style similar to a research paper",
    target_length=200,  # Reduce this number when trying out
)
bo.print_plan(plan)

blog_sections = bo.write_blog(plan.topic_analysis, plan.sections)
bo.print_blog(plan)

logger.info("Reviewing the draft")
review_response = bo.review_post(plan.topic_analysis, plan)
bo.print_review(review_response)
