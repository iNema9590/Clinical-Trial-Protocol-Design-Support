import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
model = AutoModel.from_pretrained(EMBED_MODEL).eval()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed(texts):
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


TARGET_QUERIES = {
    "objectives and endpoints": """
    clinical trial objectives primary objective secondary objective exploratory objective
    primary endpoint secondary endpoint exploratory endpoint
    efficacy endpoint safety endpoint outcome measures
    estimand analysis population
    objectives and endpoints section
    """,
    "eligibility": """
    inclusion criteria exclusion criteria eligibility criteria
    eligible subjects study population screening criteria
    inclusion requirements exclusion requirements
    criteria for participation subject eligibility
    """,
    "schedule of activities": """
    schedule of activities study flow table visit schedule
    visit schedule table study visits timepoints
    assessment schedule procedures by visit
    screening visit treatment visit follow-up visit
    study day visit window
    """,
    "visit_definitions": """
    visit definitions visit schedule description
    screening visit day 1 day 29 follow-up visit
    illness visit safety follow-up early termination visit
    visit window timing of visits study flow
    conditional visits triggered visits
    """
}


def classify_sections(sections: dict):

    section_ids = list(sections.keys())
    section_texts = [
        sections[sid]["title"] + " " + sections[sid]["content"][:20]
        for sid in section_ids
    ]

    section_embeddings = embed(section_texts)

    query_embeddings = embed(list(TARGET_QUERIES.values()))

    routing = {k: [] for k in TARGET_QUERIES}

    for i, sid in enumerate(section_ids):
        for j, target in enumerate(TARGET_QUERIES.keys()):
            score = torch.matmul(
                section_embeddings[i],
                query_embeddings[j]
            ).item()

            if score > 0.4:
                routing[target].append((sid, score))

    for k in routing:
        routing[k] = sorted(routing[k], key=lambda x: x[1], reverse=True)[:3]

    return routing
