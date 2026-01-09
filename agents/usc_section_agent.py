# agents/usc_section_agent.py

from crewai import Agent, LLM
from tools.usc_sections_search_tool import search_usc_sections

llm = LLM(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1536  # keep outputs compact; UI needs summaries, not walls of text
)

usc_section_agent = Agent(
    role="USC Statute Retriever",
    goal=(
        "Use the USC Sections Search Tool to return 3–5 unique, highly relevant statutes "
        "and short excerpts supported by the tool output."
    ),
    backstory=(
        "You are a statute retrieval assistant. You do not invent citations, titles, or text. "
        "You keep excerpts short and only describe relevance in one sentence grounded in the user facts."
    ),
    tools=[search_usc_sections],
    llm=llm,
    verbose=False,
)
