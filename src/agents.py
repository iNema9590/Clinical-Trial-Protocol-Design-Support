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
    Extract inclusion and exclusion criteria as validated EligibilityCriteria schema with structured rules.
    """
    prompt = f"""You are a clinical trial protocol analysis expert.

Your task is to extract ALL inclusion and ALL exclusion criteria from the protocol text below.

For EACH criterion, you must extract:
1. text: The exact criterion text from the protocol
2. field: The patient data field name if it can be machine-evaluated, otherwise null
3. operator: The comparison operator if evaluable, otherwise null
4. value: The threshold value if evaluable, otherwise null
5. evaluable: ONLY true if the criterion maps to an available dataset field AND can be automatically evaluated

IMPORTANT: 
- Extract ALL criteria including non-evaluable ones
- Do NOT skip criteria just because they're not machine-readable
- Mark evaluable=false for criteria requiring human judgment or clinical assessment
- Only set evaluable=true if all of: field exists in dataset, operator is clear, value can be determined

AVAILABLE CSV FIELD NAMES (case-sensitive):
Demographic & Lab:
- AGE - numeric, patient age in years
- BMI - numeric, body mass index kg/m²
- ALT - numeric, alanine aminotransferase level (U/L)
- ULN - numeric constant at 40 (upper limit of normal for ALT)

Test Results:
- PCR_RESULT - string, values: "Positive" or "Negative"

Clinical Flags:
- SEVERE_ALLERGY_HISTORY - boolean, "True" or "False"
- IMMUNOSUPPRESSIVE_THERAPY_6M - boolean, "True" or "False"
- PREGNANT - boolean, "True" or "False"
- GUILLAIN_BARRE_HISTORY - boolean, "True" or "False"
- IMMUNODEFICIENCY_CONDITION - boolean, "True" or "False"
- MALIGNANCY_HISTORY - boolean, "True" or "False"
- BLEEDING_DISORDER_HISTORY - boolean, "True" or "False"
- SEVERE_COMORBIDITY - boolean, "True" or "False"
- INVESTIGATIONAL_SARS_COV2_DRUG - boolean, "True" or "False"
- RECENT_IMMUNOGLOBULIN_3M - boolean, "True" or "False"
- STUDY_STAFF_INVOLVEMENT - boolean, "True" or "False"

Clinical Assessment:
- BODY_TEMPERATURE - numeric, in Celsius (e.g., 37.2, 38.5)
- RECENT_VACCINE_30D - boolean, "True" or "False" (vaccine within 30 days other than flu)
- RECENT_BLOOD_DONATION_30D - boolean, "True" or "False" (blood donation within 30 days)

Medical Status & Judgment:
- SARS_COV2_RISK_LEVEL - categorical, "Low" or "High" (SARS-CoV-2 infection risk)
- MEDICAL_STABILITY_STATUS - categorical, "Stable" or "Unstable" (hospitalization risk assessment)
- COGNITIVE_COMPLIANCE_CAPABLE - boolean, "True" or "False" (ability to understand/comply)
- USING_CONTRACEPTION - boolean, "True" or "False" (contraceptive use by women)
- CONSENTED - boolean, "True" or "False" (informed consent given)

FIELD MAPPING FROM PROTOCOL LANGUAGE:
- "Age"/"aged"/"years old" → "AGE"
- "BMI"/"body mass index" → "BMI"
- "PCR"/"SARS-CoV-2"/"COVID" → "PCR_RESULT"
- "ALT"/"liver enzyme"/"transaminase" → "ALT"/"ULN"
- "allergy"/"severe allergy" → "SEVERE_ALLERGY_HISTORY"
- "immunosuppressive"/"immunosuppression" → "IMMUNOSUPPRESSIVE_THERAPY_6M"
- "pregnant"/"pregnancy"/"pregnancy test" → "PREGNANT"
- "fever"/"elevated temperature" → "BODY_TEMPERATURE"
- "vaccine"/"vaccination" → "RECENT_VACCINE_30D"
- "blood donation" → "RECENT_BLOOD_DONATION_30D"
- "Guillain-Barré"/"demyelinating condition" → "GUILLAIN_BARRE_HISTORY"
- "immunosuppressive state"/"immunodeficient" → "IMMUNODEFICIENCY_CONDITION"
- "primary malignancy"/"cancer history" → "MALIGNANCY_HISTORY"
- "bleeding disorder"/"coagulopathy"/"significant bleeding" → "BLEEDING_DISORDER_HISTORY"
- "severe"/"uncontrolled"/"cardiovascular disease"/"respiratory disease" → "SEVERE_COMORBIDITY"
- "investigational products"/"COVID-19 treatment" → "INVESTIGATIONAL_SARS_COV2_DRUG"
- "immunoglobulins"/"blood products" → "RECENT_IMMUNOGLOBULIN_3M"
- "Increased risk of SARS-CoV-2 infection" → "SARS_COV2_RISK_LEVEL" (== "High")
- "Medically stable"/"hospitalization anticipated" → "MEDICAL_STABILITY_STATUS" (== "Stable")
- "Able to understand and comply"/"able to participate" → "COGNITIVE_COMPLIANCE_CAPABLE"
- "Contraceptive use by women"/"birth control" → "USING_CONTRACEPTION"
- "Capable of giving signed informed consent" → "CONSENTED"
- "Involvement in planning"/"study staff" → "STUDY_STAFF_INVOLVEMENT"

VALUE FORMAT RULES:
- For "Positive"/"Negative": use exact case
- For boolean: use "True" or "False" (exact)
- For numeric: use plain numbers (e.g., 18, 37.8)
- For temperatures > 100°F: convert to Celsius first (>37.8°C)
- For ranges: use "min,max" format (e.g., "18,65")
- For multipliers: use format "X*ULN" (e.g., "2*ULN")

EXAMPLES OF EVALUABLE CRITERIA:
- "Age ≥ 18 years" → field: "AGE", operator: ">=", value: "18", evaluable: true
- "BMI < 35 kg/m²" → field: "BMI", operator: "<", value: "35", evaluable: true
- "Negative PCR on screening" → field: "PCR_RESULT", operator: "==", value: "Negative", evaluable: true
- "No severe allergy" → field: "SEVERE_ALLERGY_HISTORY", operator: "==", value: "False", evaluable: true
- "Body temperature ≤ 37.8°C" → field: "BODY_TEMPERATURE", operator: "<=", value: "37.8", evaluable: true
- "No vaccine within 30 days" → field: "RECENT_VACCINE_30D", operator: "==", value: "False", evaluable: true
- "ALT ≤ 2 × ULN" → field: "ALT", operator: "<=", value: "2*ULN", evaluable: true

EXAMPLES OF NON-EVALUABLE CRITERIA:
- "Medically stable and unlikely to require hospitalization" → field: null, operator: null, value: null, evaluable: false
- "Able to understand and comply with study requirements" → field: null, operator: null, value: null, evaluable: false
- "History of Guillain-Barré syndrome" → field: null, operator: null, value: null, evaluable: false
- "Significant infection or other acute illness" → field: null, operator: null, value: null, evaluable: false
- "Any other significant disease or disorder" → field: null, operator: null, value: null, evaluable: false
- "Involvement in planning or conduct of the study" → field: null, operator: null, value: null, evaluable: false

OPERATOR RULES:
- Use "==" for equality (including "Positive", "Negative", "True", "False")
- Use "!=" for inequality/exclusion (e.g., PCR must NOT be positive)
- Use "between" for age/BMI ranges
- Use ">=", "<=", ">", "<" for numeric thresholds

Examples:
- "Age between 18 and 65 years" → {{"text": "Age between 18 and 65 years", "field": "AGE", "operator": "between", "value": "18,65", "evaluable": true}}
- "BMI ≥ 18.5 kg/m²" → {{"text": "BMI ≥ 18.5 kg/m²", "field": "BMI", "operator": ">=", "value": "18.5", "evaluable": true}}
- "BMI < 35" → {{"text": "BMI < 35", "field": "BMI", "operator": "<", "value": "35", "evaluable": true}}
- "Positive PCR test result" → {{"text": "Positive PCR test result", "field": "PCR_RESULT", "operator": "==", "value": "Positive", "evaluable": true}}
- "Negative PCR test" → {{"text": "Negative PCR test", "field": "PCR_RESULT", "operator": "==", "value": "Negative", "evaluable": true}}
- "No positive PCR" → {{"text": "No positive PCR", "field": "PCR_RESULT", "operator": "!=", "value": "Positive", "evaluable": true}}
- "ALT ≤ 2 × ULN" → {{"text": "ALT ≤ 2 × ULN", "field": "ALT", "operator": "<=", "value": "2*ULN", "evaluable": true}}
- "ALT less than 3 times ULN" → {{"text": "ALT less than 3 times ULN", "field": "ALT", "operator": "<", "value": "3*ULN", "evaluable": true}}
- "History of severe allergies" → {{"text": "History of severe allergies", "field": "SEVERE_ALLERGY_HISTORY", "operator": "==", "value": "True", "evaluable": true}}
- "No severe allergy history" → {{"text": "No severe allergy history", "field": "SEVERE_ALLERGY_HISTORY", "operator": "==", "value": "False", "evaluable": true}}
- "No immunosuppressive therapy in past 6 months" → {{"text": "No immunosuppressive therapy in past 6 months", "field": "IMMUNOSUPPRESSIVE_THERAPY_6M", "operator": "==", "value": "False", "evaluable": true}}
- "Willing to provide informed consent" → {{"text": "Willing to provide informed consent", "field": null, "operator": null, "value": null, "evaluable": false}}

Rules:
- Extract exact wording from the protocol in "text" field
- Mark evaluable=false for ANY criteria that cannot be automatically evaluated
- Do NOT skip criteria just because they're non-evaluable
- Do NOT invent criteria
- Each criterion is a separate object in the array
- Use EXACT field names, operators, and value formats as specified above

Return ONLY this JSON structure:
{{
  "inclusion": [
    {{"text": "Adult, ≥ 18 years of age", "field": "AGE", "operator": ">=", "value": "18", "evaluable": true}},
    {{"text": "Medically stable and unlikely to need hospitalization", "field": null, "operator": null, "value": null, "evaluable": false}},
    {{"text": "Able to understand and comply with requirements", "field": null, "operator": null, "value": null, "evaluable": false}}
  ],
  "exclusion": [
    {{"text": "History of allergy to vaccine component", "field": "SEVERE_ALLERGY_HISTORY", "operator": "==", "value": "True", "evaluable": true}},
    {{"text": "History of Guillain-Barré syndrome", "field": null, "operator": null, "value": null, "evaluable": false}},
    {{"text": "Fever > 100°F on day prior to randomization", "field": "BODY_TEMPERATURE", "operator": ">", "value": "37.8", "evaluable": true}}
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
