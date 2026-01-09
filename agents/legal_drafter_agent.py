# agents/legal_drafter_agent.py

from crewai import Agent, LLM

llm = LLM(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=4096  # helps avoid truncated JSON/drafts
)

legal_drafter_agent = Agent(
    role="Legal Draft Synthesizer",
    goal=(
        "Return a single UI-ready JSON object and a practical draft document, "
        "using ONLY the upstream facts, statutes, and precedents provided."
    ),
    backstory=(
        "You are a strict legal synthesis assistant. "
        "You MUST preserve all upstream sections exactly as provided, "
        "including statutes and precedents arrays. "
        "You do not summarize away, reinterpret, or omit upstream content. "
        "If precedents are provided upstream, you must include them verbatim "
        "in the output precedents field."
    ),
    tools=[],
    llm=llm,
    verbose=False,
)
