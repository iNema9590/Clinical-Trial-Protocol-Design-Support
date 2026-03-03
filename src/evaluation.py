"""
LLM-as-Judge evaluation for agent outputs.
"""
import json
import re
from typing import Dict, Any, Optional
from llm import generate


def llm_as_judge_evaluation(
    full_document: str,
    agent_prompt: str,
    agent_output: str,
    agent_name: str
) -> Dict[str, Any]:
    """
    Evaluate agent output using LLM-as-judge approach.
    
    Args:
        full_document: The complete protocol document text
        agent_prompt: The prompt used by the agent (without the document)
        agent_output: The agent's output to evaluate
        agent_name: Name of the agent being evaluated
        
    Returns:
        Dictionary with evaluation results including score, reasoning, and completeness
    """
    
    evaluation_prompt = f"""You are an expert evaluator assessing the quality of AI agent outputs for clinical trial protocol analysis.

CONTEXT:
You are evaluating the output of an AI agent called "{agent_name}".
The agent was given the following task/prompt:

{agent_prompt}

FULL PROTOCOL DOCUMENT (Ground Truth):
\"\"\"
{full_document}
\"\"\"

AGENT'S OUTPUT:
\"\"\"
{agent_output}
\"\"\"

EVALUATION TASK:
Evaluate how well the agent's output fulfills the requirements specified in the prompt, given the full protocol document as ground truth.

Provide your evaluation in the following JSON format:

{{
  "overall_score": <float between 0-10>,
  "completeness_score": <float between 0-10>,
  "accuracy_score": <float between 0-10>,
  "format_compliance_score": <float between 0-10>,
  "reasoning": "<detailed explanation of your evaluation>",
  "strengths": ["<strength 1>", "<strength 2>", ...],
  "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
  "missing_items": ["<item 1>", "<item 2>", ...],
  "incorrect_items": ["<item 1>", "<item 2>", ...],
  "suggestions": "<suggestions for improvement>"
}}

SCORING GUIDELINES:

Completeness (0-10):
- 10: All relevant information extracted, nothing missed
- 7-9: Most information extracted, minor omissions
- 4-6: Significant information missing
- 0-3: Major information missing

Accuracy (0-10):
- 10: All information accurate, no fabrications
- 7-9: Minor inaccuracies or interpretation issues
- 4-6: Several inaccuracies present
- 0-3: Major inaccuracies or fabricated information

Format Compliance (0-10):
- 10: Perfect adherence to required output structure
- 7-9: Minor format issues
- 4-6: Significant format deviations
- 0-3: Major format violations

Overall Score (0-10):
- Average of the three component scores
- Consider relative importance based on the task

IMPORTANT:
- Be strict but fair in your evaluation
- Reference specific examples from the document when noting missing or incorrect items
- Consider whether omissions are meaningful or trivial
- Output ONLY valid JSON, no additional text

Return your evaluation:
"""
    
    output = generate(evaluation_prompt)
    
    # Parse the JSON response
    try:
        # Remove markdown code fences if present
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()
        
        evaluation = json.loads(output)
        evaluation["agent_name"] = agent_name
        evaluation["success"] = True
        
    except json.JSONDecodeError as e:
        evaluation = {
            "agent_name": agent_name,
            "success": False,
            "error": f"Failed to parse JSON: {str(e)}",
            "raw_output": output
        }
    
    return evaluation


def print_evaluation_report(evaluation: Dict[str, Any]):
    """
    Pretty print the evaluation results.
    
    Args:
        evaluation: Dictionary with evaluation results
    """
    
    if not evaluation.get("success", True):
        print(f"❌ Evaluation failed: {evaluation.get('error')}")
        return
    
    agent_name = evaluation.get("agent_name", "Unknown")
    
    print("="*80)
    print(f"EVALUATION REPORT: {agent_name.upper()}")
    print("="*80)
    print()
    
    # Scores
    print("SCORES:")
    print(f"  Overall:            {evaluation['overall_score']:.1f}/10")
    print(f"  Completeness:       {evaluation['completeness_score']:.1f}/10")
    print(f"  Accuracy:           {evaluation['accuracy_score']:.1f}/10")
    print(f"  Format Compliance:  {evaluation['format_compliance_score']:.1f}/10")
    print()
    
    # Reasoning
    print("REASONING:")
    print(f"  {evaluation['reasoning']}")
    print()
    
    # Strengths
    if evaluation.get('strengths'):
        print("STRENGTHS:")
        for strength in evaluation['strengths']:
            print(f"  ✓ {strength}")
        print()
    
    # Weaknesses
    if evaluation.get('weaknesses'):
        print("WEAKNESSES:")
        for weakness in evaluation['weaknesses']:
            print(f"  ✗ {weakness}")
        print()
    
    # Missing items
    if evaluation.get('missing_items'):
        print("MISSING ITEMS:")
        for item in evaluation['missing_items']:
            print(f"  • {item}")
        print()
    
    # Incorrect items
    if evaluation.get('incorrect_items'):
        print("INCORRECT ITEMS:")
        for item in evaluation['incorrect_items']:
            print(f"  • {item}")
        print()
    
    # Suggestions
    if evaluation.get('suggestions'):
        print("SUGGESTIONS FOR IMPROVEMENT:")
        print(f"  {evaluation['suggestions']}")
        print()
    
    print("="*80)


def evaluate_all_agents(
    full_document: str,
    agent_outputs: Dict[str, Any],
    prompts: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple agent outputs.
    
    Args:
        full_document: The complete protocol document
        agent_outputs: Dict mapping agent names to their outputs
        prompts: Dict mapping agent names to their prompts
        
    Returns:
        Dict mapping agent names to evaluation results
    """
    
    evaluations = {}
    
    for agent_name, output in agent_outputs.items():
        if agent_name not in prompts:
            print(f"Warning: No prompt found for agent '{agent_name}', skipping...")
            continue
        
        print(f"Evaluating {agent_name}...")
        
        # Convert output to string if it's a Pydantic model
        if hasattr(output, 'model_dump_json'):
            output_str = output.model_dump_json(indent=2)
        elif isinstance(output, dict):
            output_str = json.dumps(output, indent=2)
        else:
            output_str = str(output)
        
        evaluation = llm_as_judge_evaluation(
            full_document=full_document,
            agent_prompt=prompts[agent_name],
            agent_output=output_str,
            agent_name=agent_name
        )
        
        evaluations[agent_name] = evaluation
    
    return evaluations


def print_summary_report(evaluations: Dict[str, Dict[str, Any]]):
    """
    Print a summary table of all evaluations.
    
    Args:
        evaluations: Dict mapping agent names to evaluation results
    """
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Agent':<30} {'Overall':>10} {'Complete':>10} {'Accurate':>10} {'Format':>10}")
    print("-"*80)
    
    for agent_name, eval_result in evaluations.items():
        if eval_result.get("success", True):
            print(f"{agent_name:<30} "
                  f"{eval_result['overall_score']:>9.1f} "
                  f"{eval_result['completeness_score']:>9.1f} "
                  f"{eval_result['accuracy_score']:>9.1f} "
                  f"{eval_result['format_compliance_score']:>9.1f}")
        else:
            print(f"{agent_name:<30} {'ERROR':>10}")
    
    print("="*80)


def extract_agent_prompts() -> Dict[str, str]:
    """
    Extract the core prompt text from each agent function (without the content insertion).
    
    Returns:
        Dict mapping agent names to their prompt templates
    """
    
    prompts = {
        "objectives and endpoints": """You are a clinical trial protocol analysis expert.

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

Expected output: JSON with primary, secondary, exploratory, and other objectives, each with their endpoints.""",

        "eligibility": """You are a clinical trial protocol analysis expert.

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

Expected output: JSON with inclusion and exclusion lists, each criterion with text, field, operator, value, and evaluable flag.""",

        "schedule of activities": """You are a clinical trial protocol analysis expert.

Your task is to extract ALL Schedule of Activities (SoA) tables from the protocol text below.

Strict Instructions:
1. Each table must be processed independently.
2. Do NOT merge visits from different tables.
3. Preserve table separation.
4. Output MUST be ONE valid JSON object.
5. Do NOT output multiple root JSON objects.
6. Do NOT include commentary.

Expected output: JSON with tables array, each containing table_title, and visits with visit_name, study_day, window, and procedures.""",

        "visit definitions": """You are a clinical trial protocol analysis expert.

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

Expected output: JSON with visits array, each containing name, description, timing, window, and trigger.""",

        "key assessments": """You are a clinical trial protocol analysis expert.

Your task is to extract all key assessments and their associated procedures from the protocol text below.

Definitions:

Assessment:
A high-level evaluation defined in the protocol (e.g., Safety Assessment, Tumor Response Assessment, Laboratory Assessment).

Procedure:
A specific test, measurement, or action performed as part of an assessment.

For EACH assessment, extract:
- category: Category explicitly stated in the protocol (e.g., "Safety", "Efficacy", "Laboratory").
- name: The exact assessment name as written in the protocol.
- description: A brief description of the assessment (1–3 sentences).
- procedures: A list of procedures belonging to this assessment.

For EACH procedure, extract:
- name: The exact procedure name as written in the protocol.
- description: A brief description (1–2 sentences).

Important Rules:
1. Preserve exact wording for names.
2. Do NOT invent assessments or procedures.
3. Do NOT include visit timing or Schedule of Activities information.
4. Do NOT group assessments by category at the top level.
5. Return assessments as a LIST under the key "assessments".

Expected output: JSON with assessments array, each containing category, name, description, and procedures."""
    }
    
    return prompts
