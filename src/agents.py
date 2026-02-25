from llm import generate


def extract_objectives(content: str) -> str:

    prompt = f"""
You are a clinical trial protocol analysis expert.

Your task is to extract ALL study objectives and ALL endpoints described in the protocol text below.

Carefully identify:

1. Primary objectives
2. Secondary objectives
3. Exploratory objectives (if any)

For each objective, extract:
- The objective statement (as written)
- The corresponding endpoint(s)
- Whether the endpoint is primary, secondary, or exploratory
- Any time frame associated with the endpoint
- Any population specification (e.g., ITT, per-protocol, safety population)

Important instructions:
- Objectives and endpoints may appear in both narrative text and tables.
- Endpoints may be described as outcome measures.
- Include efficacy and safety endpoints.
- Do NOT invent information.
- If something is not explicitly stated, do not infer it.
- Preserve original wording as much as possible.

You MUST return your response as valid JSON ONLY. Do not include any text before or after the JSON.

Use the following JSON structure:
{{
  "primary_objectives": [
    {{
      "objective": "objective text",
      "endpoints": ["endpoint 1", "endpoint 2"],
      "time_frame": "time frame if stated",
      "population": "population if stated"
    }}
  ],
  "secondary_objectives": [
    {{
      "objective": "objective text",
      "endpoints": ["endpoint 1"],
      "time_frame": "time frame if stated",
      "population": "population if stated"
    }}
  ],
  "exploratory_objectives": [
    {{
      "objective": "objective text",
      "endpoints": ["endpoint 1"],
      "time_frame": "time frame if stated",
      "population": "population if stated"
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


def extract_eligibility(content: str) -> str:
    """
    Extract inclusion and exclusion criteria as structured JSON.
    """
    prompt = f"""
    You are a clinical trial protocol analysis expert.

    Your task is to extract ALL inclusion criteria and ALL exclusion criteria from the protocol text below.

    Carefully identify:

    • Every inclusion criterion
    • Every exclusion criterion

    Important instructions:

    - Criteria are usually presented as numbered or bulleted lists.
    - Preserve the original wording as much as possible.
    - Do NOT summarize or paraphrase.
    - Do NOT merge multiple criteria into one.
    - Do NOT invent or infer criteria that are not explicitly stated.
    - If inclusion and exclusion criteria appear in the same section, separate them correctly.
    - If criteria are divided into subsections (e.g., general inclusion, COVID-specific exclusion), preserve them under the correct category.

    You MUST return your response as valid JSON ONLY. Do not include any text before or after the JSON.

    Use the following JSON structure:
    {{
      "inclusion_criteria": [
        "criterion 1",
        "criterion 2",
        "criterion 3"
      ],
      "exclusion_criteria": [
        "criterion 1",
        "criterion 2",
        "criterion 3"
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

def extract_soa(content: str) -> str:

    prompt = f"""
        You are a clinical trial protocol analysis expert.

        Your task is to extract and reconstruct the full Schedule of Activities (SoA) from the protocol text below.

        The Schedule of Activities describes:

        • Study visits (e.g., Screening, Day 1, Day 29, Follow-up, Illness Visit, etc.)
        • Visit timing and visit windows (e.g., ±3 days)
        • Procedures and assessments performed at each visit
        • Dosing or intervention administration timepoints
        • Sample collection timepoints (e.g., blood draws, swabs)

        Important instructions:

        - The SoA is usually presented as one or more tables.
        - Carefully reconstruct the visit schedule in a structured and readable format.
        - Preserve visit names exactly as written.
        - Include visit timing (study day, week, month, and window if provided).
        - For each visit, list all procedures performed at that visit.
        - If procedures are marked with symbols (e.g., X, ✓), interpret them as “performed”.
        - Do NOT summarize.
        - Do NOT omit visits.
        - Do NOT invent visits or procedures not explicitly shown.

        You MUST return your response as valid JSON ONLY. Do not include any text before or after the JSON.

        Use the following JSON structure:
        {{
          \"schedule_of_activities\": [
            {{
              \"visit_name\": \"Visit Name\",
              \"timing\": \"timing description\",
              \"window\": \"window if stated\",
              \"procedures\": [
                \"procedure 1\",
                \"procedure 2\",
                \"procedure 3\"
              ]
            }}
          ]
        }}

        If multiple SoA tables exist (e.g., main study, illness visits, safety follow-up), include all visits in the array.

        Protocol text:
        \"\"\"
        {content}
        \"\"\"

        Return ONLY valid JSON:
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