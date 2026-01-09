# tasks/legal_precedent_task.py
from crewai import Task
from agents.legal_precedent_agent import legal_precedent_agent
from tasks.case_intake_task import case_intake_task
from tasks.usc_section_task import usc_section_task

legal_precedent_task = Task(
    agent=legal_precedent_agent,
    context=[case_intake_task, usc_section_task],
    description=(
        "Return VALID JSON ONLY (double quotes). No markdown, no extra text.\n\n"

        "Goal: Find up to 3 relevant U.S. FEDERAL judicial precedents using the Legal Precedent Search Tool.\n\n"

        "MANDATORY TOOL USE:\n"
        "- You MUST call the Legal Precedent Search Tool at least once.\n"
        "- If you do not call the tool, you must return precedents=[] and explain why in notes.\n\n"

        "INPUTS AVAILABLE:\n"
        "- From intake: primary_issue, key_facts, federal_hooks.\n"
        "- From USC task: top_statutes[].citation (may be empty).\n\n"

        "HOW TO SEARCH (build 2–3 queries total):\n"
        "1) Use primary_issue + 1 USC citation (if available).\n"
        "2) Use primary_issue + 'Supreme Court' (if issue is broad/important).\n"
        "3) Optional: use primary_issue + 'circuit' or 'CourtListener opinion'.\n\n"

        "Query templates (pick what fits):\n"
        '  - "<primary_issue> <USC citation> Supreme Court"\n'
        '  - "<primary_issue> <USC citation> circuit opinion"\n'
        '  - "<primary_issue> federal case opinion CourtListener"\n'
        '  - "<primary_issue> <USC citation> site:law.cornell.edu/supremecourt/text"\n'
        '  - "<primary_issue> <USC citation> site:law.justia.com/cases"\n\n'

        "SELECTION RULES (STRICT):\n"
        "- Prefer Supreme Court, then Circuit Courts.\n"
        "- Only accept results that clearly look like a CASE / OPINION page.\n"
        "  Minimum signal: title/snippet has 'v.' OR a reporter-like string (e.g., 'U.S.', 'S. Ct.', 'F.3d', 'F. Supp.').\n"
        "- Exclude obvious non-opinion materials (briefs, amicus, petitions, dockets, law review articles).\n"
        "- Avoid PDFs unless the URL/title clearly indicates it is the opinion text from a trusted source\n"
        "  (e.g., law.cornell.edu/supct/pdf, CourtListener opinion, cases.justia.com).\n\n"

        "NO-GUESSING RULE (CRITICAL):\n"
        "- Do NOT invent/guess court, year, citation, or holding.\n"
        "- court_year MUST be supported by title/snippet/url; otherwise set \"\".\n"
        "- citation MUST be supported by title/snippet/url; otherwise set \"\".\n"
        "- holding must be 1 conservative sentence:\n"
        "  * If outcome/rule is not explicit in snippet/title: 'Addresses interpretation of <issue> under <statute>.'\n"
        "  * If snippet/title states a clear outcome: summarize in <= 1 sentence.\n\n"

        "DEDUPING:\n"
        "- Deduplicate by case name (case-insensitive). Keep the best/most official-looking source.\n"
        "- Keep <= 3 precedents.\n"
        "- If results are weak/unrelated: precedents=[].\n\n"

        "OUTPUT SCHEMA (exactly these keys; no others):\n"
        "{\n"
        '  \"precedents\": [\n'
        "    {\n"
        '      \"name\": \"string\",\n'
        '      \"court_year\": \"string\",\n'
        '      \"citation\": \"string or empty\",\n'
        '      \"holding\": \"string (<= 1 sentence, conservative)\",\n'
        '      \"relevance\": \"string (<= 1 sentence, tied to user facts)\",\n'
        '      \"url\": \"string or empty\"\n'
        "    }\n"
        "  ],\n"
        '  \"notes\": [\"string\"]\n'
        "}\n\n"

        "NOTES:\n"
        "- If precedents=[] include a short reason (e.g., 'No high-confidence federal case opinion pages found from trusted sources.').\n"
        "- If you included cases but court/year/citation are blank due to limited snippets, say so.\n"
    ),
    expected_output='{"precedents":[],"notes":["No high-confidence federal case opinion pages found from trusted sources for this issue."]}',
)
