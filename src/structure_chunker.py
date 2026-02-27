import re
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class StructuredChunk:
    chunk_id: str
    section_id: str
    level: int
    title: str
    full_title: str
    parent_id: str
    content: str


SECTION_REGEX = re.compile(
    r'(?<=\n)(?P<number>\d+)\s{2,}(?P<title>[A-Z][A-Z\s\-()]+)(?=\n)'
)

SUBSECTION_REGEX = re.compile(
    r'(?<=\n)(?P<number>\d+\.\d+(?:\.\d+)*)\s{2,}(?P<title>[A-Z][A-Za-z0-9\s\-():/]+)(?=\n)'
)


def extract_headers(text: str):

    headers = []

    for match in SECTION_REGEX.finditer(text):
        headers.append({
            "number": match.group("number"),
            "title": match.group("title").strip(),
            "start": match.start(),
            "end": match.end()
        })

    for match in SUBSECTION_REGEX.finditer(text):
        headers.append({
            "number": match.group("number"),
            "title": match.group("title").strip(),
            "start": match.start(),
            "end": match.end()
        })

    headers = sorted(headers, key=lambda x: x["start"])
    return headers


def build_structured_chunks(text: str) -> List[StructuredChunk]:

    headers = extract_headers(text)
    if not headers:
        return []

    # Map number → title
    header_map = {h["number"]: h["title"] for h in headers}

    chunks = []

    def has_child(idx):
        if idx >= len(headers) - 1:
            return False
        current = headers[idx]["number"]
        nxt = headers[idx + 1]["number"]
        return nxt.startswith(current + ".")

    for i, header in enumerate(headers):

        # Only leaf nodes
        if has_child(i):
            continue

        start = header["start"]
        end = headers[i + 1]["start"] if i + 1 < len(headers) else len(text)

        content = text[start:end].strip()

        section_id = header["number"]
        level = section_id.count(".") + 1

        # Build parent hierarchy
        parts = section_id.split(".")
        parent_id = ".".join(parts[:-1]) if len(parts) > 1 else None

        ancestors = []
        for j in range(1, len(parts)):
            ancestor_num = ".".join(parts[:j])
            if ancestor_num in header_map:
                ancestors.append(header_map[ancestor_num])

        full_title = " > ".join(ancestors + [header["title"]])

        chunk = StructuredChunk(
            chunk_id=f"sec_{section_id}",
            section_id=section_id,
            level=level,
            title=header["title"],
            full_title=full_title,
            parent_id=parent_id,
            content=content
        )

        chunks.append(chunk)

    return chunks