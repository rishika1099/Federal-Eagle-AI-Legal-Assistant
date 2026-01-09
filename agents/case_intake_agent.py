# agents/case_intake_agent.py

from crewai import Agent, LLM

llm = LLM(
    model="gpt-4o-mini",
    temperature=0
)

case_intake_agent = Agent(
    role="Case Intake",
    goal=(
        "Convert the user's description into a concise, structured intake summary "
        "for downstream statute retrieval and drafting."
    ),
    backstory=(
        "You extract facts and classify the issue. You avoid legal advice and do not draft filings."
    ),
    llm=llm,
    tools=[],
    verbose=False,  # reduce console noise
)
