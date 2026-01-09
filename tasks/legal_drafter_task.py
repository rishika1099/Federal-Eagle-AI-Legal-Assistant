# tasks/legal_drafter_task.py
from crewai import Task
from agents.legal_drafter_agent import legal_drafter_agent
from tasks.case_intake_task import case_intake_task
from tasks.usc_section_task import usc_section_task
from tasks.legal_precedent_task import legal_precedent_task

legal_drafter_task = Task(
    agent=legal_drafter_agent,
    context=[case_intake_task, usc_section_task, legal_precedent_task],
    description=(
        "Return VALID JSON ONLY (double quotes). No markdown, no code fences, no extra text.\n"
        "Return exactly these top-level keys (no others):\n"
        "summary, statutes, elements_analysis, precedents, next_steps, clarifying_questions, draft_document, disclaimer\n\n"

        "You will receive THREE upstream JSON objects:\n"
        "1) case_intake_task JSON with: case_type, primary_issue, summary, key_facts, federal_hooks, "
        "missing_info_questions, relevant_entities, locations, dates, search_queries\n"
        "2) usc_section_task JSON with: query_used[], top_statutes[]\n"
        "3) legal_precedent_task JSON with: precedents[], notes[]\n\n"

        "HARD RULES:\n"
        "- No chain-of-thought.\n"
        "- Statutes: ONLY use usc_section_task.top_statutes as your source of citations/excerpts/metadata.\n"
        "- Do NOT invent citations or statute text.\n"
        "- Use 3–5 unique statutes max (dedupe by citation).\n"
        "- Statute excerpt must be copied/trimmed from provided excerpt/content and be <= 600 chars.\n"
        "- Disclaimer is UI-only and MUST NOT appear inside draft_document.content.\n\n"

        "PRECEDENTS (IMPORTANT):\n"
        "- You will receive legal_precedent_task JSON with precedents[] and notes[].\n"
        "- If legal_precedent_task.precedents has 1+ items, you MUST carry them into your output precedents (up to 3).\n"
        "- You are NOT allowed to output precedents=[] if upstream precedents is non-empty.\n"
        "- Only drop a precedent if it is clearly unrelated to the user’s issue; if you drop one, explain why in clarifying_questions.\n"
        "- If upstream precedents is empty, output precedents=[].\n\n"

        "CRITICAL FORMATTING RULES FOR draft_document.content:\n"
        "- ABSOLUTELY NO MARKDOWN. Do not use **, *, -, •, or numbered markdown lists.\n"
        "- Use ONLY plain text with line breaks.\n"
        "- Use ALL CAPS headings (e.g., INCIDENT OVERVIEW) and numbered paragraphs (1., 2., 3.).\n"
        "- Do NOT repeat the document title as the first line of content.\n"
        "  (The UI already shows draft_document.document_type as the title.)\n"
        "- Use placeholders like [NAME], [DATE], [LOCATION], [DISTRICT].\n\n"

        "OUTPUT SCHEMA (must match types):\n"
        'summary: {"case_type":"criminal|civil|administrative|unclear","primary_issue":"string","federal_hook":["string"],'
        '"confidence":"low|medium|high","assumptions":["string"]}\n'
        'statutes: [{"citation":"string","title":"string","title_name":"string","section_title":"string",'
        '"why_relevant":"string","elements":["string"],"excerpt":"string"}]\n'
        'elements_analysis: [{"citation":"string","checklist":[{"element":"string","status":"met|unknown|not_met",'
        '"supporting_facts":["string"]}]}]\n'
        'precedents: [{"name":"string","court_year":"string","citation":"string or empty","holding":"string","relevance":"string","url":"string"}]\n'
        'next_steps: ["string"]\n'
        'clarifying_questions: ["string"]\n'
        'draft_document: {"document_type":"string","content":"string"}\n'
        'disclaimer: "string"\n\n'

        "HOW TO POPULATE:\n"
        "- summary.case_type and summary.primary_issue come directly from intake.\n"
        "- summary.federal_hook comes from intake federal_hooks.\n"
        "- confidence: high only if user facts are specific (who/what/when/where/how) AND statutes clearly match.\n"
        "- assumptions: only include assumptions you actually used; keep to 0–4.\n\n"

        "STATUTES:\n"
        "- For each statute: title/title_name/section_title/citation/excerpt must come from top_statutes.\n"
        "- why_relevant must reference the user's key_facts (not generic).\n"
        "- elements: 3–6 plain-English elements.\n\n"

        "ELEMENTS ANALYSIS:\n"
        "- Create one checklist block per included statute.\n"
        "- Each element status must be based ONLY on key_facts:\n"
        "  met = clearly supported, not_met = clearly contradicted, unknown = not enough info.\n"
        "- supporting_facts must be short phrases derived from key_facts (no invented facts).\n\n"

        "CLARIFYING QUESTIONS:\n"
        "- Use intake missing_info_questions.\n"
        "- Add at most 2 more only if critical for statute matching.\n\n"

        "NEXT STEPS (4–8):\n"
        "- Practical actions (evidence preservation, timeline, identify accounts, request logs, etc.).\n"
        "- No legal advice, no promises, no instructions to file specific documents.\n\n"

        "DRAFT DOCUMENT SELECTION:\n"
        "- If user is reporting harm / victim: Incident Report Packet or Demand Letter\n"
        "- If business/compliance: Internal Memo or Risk Assessment\n"
        "- If explicitly hypothetical/academic: Charging Memo\n"
        "- Only draft a court complaint if user explicitly asked for a complaint.\n\n"

        "DRAFT DOCUMENT CONTENT TEMPLATES (choose one; plain text only):\n"
        "A) INCIDENT REPORT PACKET content should look like:\n"
        "INCIDENT OVERVIEW\n"
        "1. Date of Incident: [DATE]\n"
        "2. Location: [LOCATION]\n"
        "3. Reporting Party: [NAME / ROLE]\n\n"
        "SUMMARY OF EVENTS\n"
        "4. [2–4 sentences summary]\n\n"
        "KEY FACTS (NUMBERED)\n"
        "5. ...\n"
        "6. ...\n\n"
        "EVIDENCE TO PRESERVE\n"
        "7. ...\n\n"
        "REQUESTED ACTION / INTERNAL NEXT STEPS\n"
        "8. ...\n\n"
        "CONTACT INFORMATION\n"
        "9. [NAME, EMAIL, PHONE]\n\n"
        "B) DEMAND LETTER content should look like:\n"
        "[DATE]\n"
        "[RECIPIENT NAME]\n"
        "[RECIPIENT ADDRESS]\n\n"
        "RE: [SUBJECT]\n\n"
        "1. ...\n\n"
        "C) INTERNAL MEMO / RISK ASSESSMENT content should look like:\n"
        "BACKGROUND\n"
        "1. ...\n\n"
        "RISK SUMMARY\n"
        "2. ...\n\n"
        "RELEVANT STATUTES (CITATIONS ONLY)\n"
        "3. 18 U.S.C. § ...\n\n"
        "RECOMMENDED ACTIONS\n"
        "4. ...\n\n"
        "Make sure the draft matches the selected document_type.\n"
    ),
    expected_output="A single valid JSON object only.",
)
