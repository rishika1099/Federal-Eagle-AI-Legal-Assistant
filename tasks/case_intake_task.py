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
        "- federal_hooks: 1–6 short strings explaining why federal law *might* apply.\n"
        "  Each hook MUST be a FACT-SPECIFIC phrase drawn from the user's scenario — not a generic category.\n"
        "  GOOD (specific):  'classified national defense documents transmitted to foreign contact',\n"
        "                    'federally insured bank robbed with firearms',\n"
        "                    'SSN used to open accounts across multiple states',\n"
        "                    '50 kg cocaine moved Texas->New York via interstate highway',\n"
        "                    'failure to report $800,000 income to IRS over 5 years',\n"
        "                    'ransom email sent across state lines'.\n"
        "  BAD (forbidden generic boilerplate — do NOT emit these exact strings):\n"
        "                    'cross-state conduct', 'federal agency involved', 'interstate communications',\n"
        "                    'wire transfers', 'federally insured bank' (alone), 'SSN/identity documents' (alone).\n"
        "  If the only available hook is generic, prefer to leave the list shorter.\n"
        "- missing_info_questions: 0–8 questions ONLY if needed to improve accuracy\n\n"
        "- search_queries: 4–6 short queries (<= 8 words each) that will work for USC search.\n"
        "  Requirements for search_queries:\n"
        "  (1) Include 1–2 statute-anchor queries ONLY when the matching statute is well-known.\n"
        "      CRITICAL — title number must be correct. Common federal title -> domain mapping:\n"
        "        Title 18: criminal procedure (fraud, theft, robbery, kidnapping, computer crime, ID theft, RICO)\n"
        "        Title 21: drug enforcement / controlled substances (e.g., 21 U.S.C. § 841, § 846, § 952)\n"
        "        Title 26: Internal Revenue Code / tax (e.g., 26 U.S.C. § 7201, § 7206, § 7202)\n"
        "        Title 31: money & finance / BSA / structuring (e.g., 31 U.S.C. § 5324, § 5313)\n"
        "        Title 15: securities / commerce; Title 8: immigration; Title 42: civil rights/health\n"
        "      Do NOT default to title 18 for drug or tax cases. If unsure of the title number, OMIT the statute-anchor query.\n"
        "  (2) Include at least 1 plain-English crime/claim label that matches statute titles:\n"
        "      e.g. 'wire fraud', 'identity theft', 'computer fraud', 'money laundering', 'tax evasion'.\n"
        "  (3) Include at least 1 context phrase tied to the facts:\n"
        "      e.g. 'employee database unauthorized access', 'email scheme to defraud'.\n"
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
