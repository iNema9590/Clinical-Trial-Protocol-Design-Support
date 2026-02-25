import json
from llm import generate
from schemas import ObjectivesByCategory, EligibilityCriteria

def extract_objectives(content: str) -> ObjectivesByCategory:
    """
    Extract study objectives and endpoints as a validated ObjectivesByCategory schema.
    """
    prompt = f"""You are a clinical trial protocol analysis expert.

Your task is to extract ALL study objectives and ALL endpoints from the protocol text below.

For EACH objective, provide:
- objective: The exact objective text from the protocol
- endpoints: List of endpoints for this objective

Categorize objectives as:
- primary: Primary study objectives
- secondary: Secondary objectives
- exploratory: Exploratory objectives  
- other: Objectives that don't fit above categories (or empty list)

Rules:
- Preserve exact wording from the protocol
- Do NOT invent objectives or endpoints
- Each objective must have at least one endpoint
- If a section has no objectives of that type, use an empty list []

Return ONLY this JSON structure (lowercase field names):
{{
  "primary": [
    {{"objective": "text", "endpoints": ["ep1", "ep2"]}}
  ],
  "secondary": [
    {{"objective": "text", "endpoints": ["ep1"]}}
  ],
  "exploratory": [],
  "other": []
}}

Protocol text:
\"\"\"
{content}
\"\"\"

Output ONLY valid JSON. Do not include explanations or commentary.
"""

    output = generate(prompt)
    # parsed = json.loads(output.strip())
    # validated = ObjectivesByCategory(**parsed)
    return output.strip()


def extract_eligibility(content: str) -> EligibilityCriteria:
    """
    Extract inclusion and exclusion criteria as validated EligibilityCriteria schema.
    """
    prompt = f"""You are a clinical trial protocol analysis expert.

Your task is to extract ALL inclusion and ALL exclusion criteria from the protocol text below.

For each criterion:
- Extract the exact text from the protocol
- Preserve original wording
- Do NOT merge or paraphrase criteria

Categorize as:
- inclusion: All inclusion criteria (or empty list)
- exclusion: All exclusion criteria (or empty list)

Rules:
- Each criterion is a separate list item
- Do NOT invent criteria
- If a section lacks criteria, use empty list []
- Preserve exact wording from the protocol

Return ONLY this JSON structure (lowercase field names):
{{
  "inclusion": [
    "criterion 1",
    "criterion 2"
  ],
  "exclusion": [
    "criterion 1",
    "criterion 2"
  ]
}}

Protocol text:
\"\"\"
{content}
\"\"\"

Output ONLY valid JSON. Do not include explanations or commentary.
"""

    output = generate(prompt)
    parsed = json.loads(output.strip())
    validated = EligibilityCriteria(**parsed)
    return validated

def extract_soa(content: str) -> str:

    prompt = f"""
You are a clinical trial protocol analysis expert.

Your task is to extract ALL Schedule of Activities (SoA) tables from the protocol text below.

Strict Instructions:

1. Extract ONLY structured tables.
2. Output MUST contain tables in valid Markdown table format.
3. Do NOT include any explanations, commentary, or text outside the tables.
4. If multiple SoA tables exist, output each table separately.
5. Preserve original column headers exactly as written.
6. Preserve visit names exactly as written.
7. Preserve procedure names exactly as written.
8. Do NOT summarize or restructure unless formatting is broken.
9. If no SoA table is present, output: NO_SOA_FOUND

Valid Markdown table example:

| Procedure | Screening | Day 1 | Day 29 |
|-----------|------------|-------|--------|
| Blood Draw | X |  | X |

Protocol Text:
\"\"\"
{content}
\"\"\"

Return only Markdown tables:
"""
    
    output = generate(prompt)
    return output.strip()


def extract_visit_definitions(content: str) -> str:
    """
    Extract visit definitions and timing from protocol text as structured JSON.
    """
    prompt = f"""
You are a clinical trial protocol analysis expert.

Your task is to extract and clearly describe all study visit definitions and visit timing rules from the protocol text below.

Focus specifically on:

• The definition and purpose of each visit (e.g., Screening, Day 1, Day 29, Follow-up, Illness Visit, Safety Follow-up)
• When each visit occurs (study day, week, month, etc.)
• Visit windows (e.g., ±3 days)
• Conditional or triggered visits (e.g., visits triggered by symptoms or positive test)
• The sequence of visits (e.g., Day 1 followed by Day 29)
• Remote vs on-site visits (if specified)
• Early termination visits (if described)

Important instructions:

- This task is NOT about listing procedures (those belong to the Schedule of Activities).
- Instead, describe how visits are defined and how timing is determined.
- Preserve original wording where important.
- Do NOT invent visit rules not explicitly stated.
- If timing windows differ between visits, specify clearly.
- If illness visits or safety follow-up visits are triggered by specific conditions, describe those triggers.

You MUST return your response as valid JSON ONLY. Do not include any text before or after the JSON.

Use the following JSON structure:
{{
  "visit_definitions": [
    {{
      "visit_name": "Visit Name",
      "purpose": "purpose description",
      "occurs_at": "timing description",
      "visit_window": "window if stated",
      "trigger": "trigger conditions if applicable",
      "followed_by": "next visit if specified",
      "visit_type": "on-site/remote if specified"
    }}
  ]
}}

Protocol text:
\"\"\"
{content}
\"\"\"

Return ONLY valid JSON:
"""

    output = generate(prompt)
    return output.strip()