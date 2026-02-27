import json
import re
from llm import generate
from schemas import *

def extract_json_from_llm_output(raw_output: str) -> dict:
    """
    Extracts JSON from LLM output that may be wrapped in ```json ... ``` fences.
    """

    # Remove markdown code fences if present
    fenced_pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(fenced_pattern, raw_output, re.DOTALL)

    if match:
        json_str = match.group(1)
    else:
        json_str = raw_output.strip()

    # Now parse
    return json.loads(json_str)

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
  "exploratory": []
}}

Protocol text:
\"\"\"
{content}
\"\"\"

Output ONLY valid JSON. Do not include explanations or commentary.
"""

    output = generate(prompt)
    parsed = extract_json_from_llm_output(output)
    validated = ObjectivesByCategory(**parsed)
    return validated


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
    parsed = extract_json_from_llm_output(output)
    validated = EligibilityCriteria(**parsed)
    return validated

def extract_soa(content: str) -> str:

    prompt = f"""
You are a clinical trial protocol analysis expert.

Your task is to extract ALL Schedule of Activities (SoA) tables from the protocol text below.

Strict Instructions:

1. Each table must be processed independently.
2. Do NOT merge visits from different tables.
3. Preserve table separation.
6. Output MUST be ONE valid JSON object.
7. Do NOT output multiple root JSON objects.
8. Do NOT include commentary.

Return something like this structure:

{{
  "tables": [
    {{
      "table_title": "Table title if present or null",
      "visits": [
        {{
          "visit_name": "Visit Name",
          "study_day": "Day X or null",
          "window": "±X days or null",
          "procedures": ["Procedure A", "Procedure B"]
        }}
      ]
    }}
  ]
}}

The JSON structure for each SoA table should be inferred from the table content.

Protocol Text:
\"\"\"
{content}
\"\"\"

Return only Markdown tables:
"""
    
    output = generate(prompt)
    # parsed = extract_json_from_llm_output(output)
    return output.strip()


def extract_visit_definitions(content: str) -> VisitDefinitionsOutput:
    """
    Extract visit definitions and timing as structured JSON.
    """

    prompt = f"""
You are a clinical trial protocol analysis expert.

Your task is to extract ALL study visit definitions and their timing rules from the protocol text below.

Definition:
A visit definition explains what a visit is and when it occurs.
This includes screening visits, dosing visits, follow-up visits, illness visits, safety visits, and early termination visits.

For EACH visit, extract:

- name: The exact visit name as written in the protocol.
- description: A brief description of the visit purpose or definition.
- timing: When the visit occurs (e.g., Day 1, Week 4, within 28 days prior to randomization).
- window: Visit window if explicitly stated (e.g., ±3 days). Otherwise null.
- trigger: Trigger condition if the visit is conditional (e.g., symptom onset). Otherwise null.

Important Rules:

1. This task is NOT about listing procedures (those belong to Schedule of Activities).
2. Do NOT include procedures performed at the visit.
3. Do NOT invent visits or timing rules.
4. Preserve wording closely to the protocol when possible.
5. If timing is not explicitly stated, set it to null.
6. If window is not stated, set it to null.
7. If trigger is not applicable, set it to null.
8. Output MUST strictly follow the JSON structure below.
9. Output ONLY valid JSON. No explanations or commentary.

Return EXACTLY this structure:

{{
  "visits": [
    {{
      "name": "Visit Name",
      "description": "Brief visit definition",
      "timing": "When it occurs or null",
      "window": "Visit window or null",
      "trigger": "Trigger condition or null"
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
    parsed = extract_json_from_llm_output(output)
    validated = VisitDefinitionsOutput(**parsed)
    return validated


def extract_key_assessments(content: str) -> KeyAssessmentsOutput:
    """
    Extract key assessments and procedures as validated KeyAssessmentsOutput schema.
    """

    prompt = f"""
You are a clinical trial protocol analysis expert.

Your task is to extract all key assessments and their associated procedures from the protocol text below.

Definitions:

Assessment:
A high-level evaluation defined in the protocol (e.g., Safety Assessment, Tumor Response Assessment, Laboratory Assessment).

Procedure:
A specific test, measurement, or action performed as part of an assessment.

For EACH assessment, extract:

- category: Category explicitly stated in the protocol (e.g., "Safety", "Efficacy", "Laboratory").
  If no category is explicitly stated, infer the most appropriate category using a single word (e.g., "safety", "efficacy", "other").

- name: The exact assessment name as written in the protocol.

- description: A brief description of the assessment (1–3 sentences).

- procedures: A list of procedures belonging to this assessment.

For EACH procedure, extract:

- name: The exact procedure name as written in the protocol.
- description: A brief description (1–2 sentences).

Important Rules:

1. Preserve exact wording for names.
2. Do NOT invent assessments.
3. Do NOT invent procedures.
4. Do NOT include visit timing or Schedule of Activities information.
5. Do NOT group assessments by category at the top level.
6. Return assessments as a LIST under the key "assessments".
7. If no assessments are found, return:
   {{
     "assessments": []
   }}
8. Output MUST be valid JSON only.
9. Do NOT include explanations or commentary.

Return EXACTLY this JSON structure:

{{
  "assessments": [
    {{
      "category": "safety",
      "name": "Assessment Name",
      "description": "Brief description",
      "procedures": [
        {{
          "name": "Procedure Name",
          "description": "Brief description"
        }}
      ]
    }}
  ]
}}

Protocol Text:
\"\"\"
{content}
\"\"\"

Return ONLY valid JSON:
"""

    output = generate(prompt)
    parsed = extract_json_from_llm_output(output)
    validated = KeyAssessmentsOutput(**parsed)
    return validated
