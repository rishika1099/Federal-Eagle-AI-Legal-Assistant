# agents/legal_precedent_agent.py

from crewai import Agent, LLM
from tools.legal_precedent_search_tool import search_legal_precedents

llm = LLM(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1200,  # keep outputs small + structured
)

legal_precedent_agent = Agent(
    role="Precedent Finder",
    goal=(
        "You MUST find federal judicial precedents by actively using the "
        "Legal Precedent Search Tool. You only return an empty list if the tool "
        "returns no usable federal case opinions."
    ),
    backstory=(
        "You are a careful legal research assistant. You do not guess citations or holdings. "
        "If results are weak or unclear, you return an empty list."
    ),
    tools=[search_legal_precedents],
    llm=llm,
    verbose=False,
    allow_delegation=False,
)
