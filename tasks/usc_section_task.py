# tasks/usc_section_task.py

from crewai import Task
from agents.usc_section_agent import usc_section_agent
from tasks.case_intake_task import case_intake_task

# IMPORTANT:
# - Do NOT import usc_section_task from itself
# - Do NOT reference usc_section_task inside its own context

usc_section_task = Task(
    agent=usc_section_agent,
    context=[case_intake_task],
    description=(
        "Return VALID JSON ONLY. No markdown, no extra text.\n"
        "Return EXACTLY this schema and no extra keys:\n"
        '{ "query_used": [...], "top_statutes": [...] }\n\n'

        "Input: prior task JSON includes search_queries[].\n"
        "Action:\n"
        "- Pick 2–4 best search_queries (short, high-signal).\n"
        "- Use the USC Sections Search Tool on each.\n"
        "- Merge results, dedupe by citation, and select the best 3–5 statutes.\n\n"

        "Hard rules:\n"
        "- Use ONLY tool-returned metadata/text. Do not invent citations or wording.\n"
        "- excerpt must be copied/trimmed from tool output (content/excerpt) and <= 600 chars.\n"
        "- why_relevant_hint must be 1 sentence tied to the user facts (no advice).\n\n"

        "Output schema:\n"
        "{\n"
        '  "query_used": ["string", "..."],\n'
        '  "top_statutes": [\n'
        "    {\n"
        '      "citation": "string",\n'
        '      "title": "string",\n'
        '      "title_name": "string",\n'
        '      "chapter": "string or number",\n'
        '      "chapter_title": "string",\n'
        '      "section_number": "string",\n'
        '      "section_title": "string",\n'
        '      "excerpt": "string (<=600 chars)",\n'
        '      "why_relevant_hint": "string (1 sentence)"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    ),
    expected_output=(
        '{"query_used":["wire fraud statute","investment scam email"],'
        '"top_statutes":[{"citation":"18 U.S.C. § 1343","title":"18","title_name":"CRIMES AND CRIMINAL PROCEDURE",'
        '"chapter":"63","chapter_title":"Mail Fraud","section_number":"1343","section_title":"Wire fraud",'
        '"excerpt":"...","why_relevant_hint":"Interstate electronic communications were allegedly used to induce victims to send money."}]}'
    ),
)
