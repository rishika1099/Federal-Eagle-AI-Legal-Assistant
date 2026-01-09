# tasks/case_intake_task.py
from crewai import Task
from agents.case_intake_agent import case_intake_agent

case_intake_task = Task(
    agent=case_intake_agent,
    description=(
        "Output VALID JSON ONLY (no markdown, no code fences, no extra text).\n"
        "Return exactly the keys listed below. Do not add any other keys.\n\n"
        "User input:\n"
        "{user_input}\n\n"
        "Task: extract facts and classify the issue for downstream statute retrieval.\n"
        "Do NOT give legal advice. Do NOT draft documents. Do NOT cite cases.\n\n"
        "Field rules:\n"
        "- case_type MUST be one of: criminal, civil, administrative, unclear\n"
        "- legal_domain: 1–3 words (e.g., 'Cybercrime', 'Fraud', 'Tax', 'Employment', 'Immigration', 'Environment')\n"
        "- primary_issue: one short sentence (<= 18 words)\n"
        "- summary: 2–4 plain-English sentences, neutral tone\n"
        "- key_facts: 5–10 short fact strings (no speculation, no advice)\n"
        "- relevant_entities: people/orgs/systems involved (strings). Use generic labels if unknown (e.g., 'Employer', 'Bank')\n"
        "- locations: list of places mentioned; if none, use ['unknown']\n"
        "- dates: list of dates/timeframes mentioned; if none, use ['unknown']\n"
        "- federal_hooks: 1–6 short strings explaining why federal law *might* apply (facts only).\n"
        "  Examples: 'interstate communications', 'federally insured bank', 'federal program funds', 'immigration status',\n"
        "  'cross-state conduct', 'federal agency involved', 'wire transfers', 'SSN/identity documents'\n"
        "- missing_info_questions: 0–8 questions ONLY if needed to improve accuracy\n\n"
        "- search_queries: 4–6 short queries (<= 8 words each) that will work for USC search.\n"
        "  Requirements for search_queries:\n"
        "  (1) Include 1–2 statute-anchor queries ONLY if truly obvious:\n"
        "      Examples: '18 U.S.C. § 1030', '18 U.S.C. § 1343', '18 U.S.C. § 1344', '18 U.S.C. § 1028A'.\n"
        "  (2) Include at least 1 plain-English crime/claim label that matches statute titles:\n"
        "      Examples: 'wire fraud', 'identity theft', 'computer fraud', 'money laundering', 'retaliation'.\n"
        "  (3) Include at least 1 context phrase tied to the facts:\n"
        "      Examples: 'employee database unauthorized access', 'email scheme to defraud', 'SSN used to open accounts'.\n"
        "  Keep queries concrete; avoid vague single words like 'fraud' alone.\n\n"
        "Return JSON with exactly these keys (same order is preferred):\n"
        "case_type, legal_domain, primary_issue, summary, key_facts, relevant_entities, locations, dates,\n"
        "federal_hooks, missing_info_questions, search_queries"
    ),
    expected_output=(
        '{"case_type":"unclear","legal_domain":"Fraud","primary_issue":"string","summary":"string",'
        '"key_facts":["..."],"relevant_entities":["..."],"locations":["unknown"],"dates":["unknown"],'
        '"federal_hooks":["..."],"missing_info_questions":["..."],"search_queries":["..."]}'
    ),
)
