import re
from typing import Dict

def split_into_sections(text: str) -> Dict[str, str]:
    """
    Split text into sections and subsections only.
    
    Subsubsections are not separate chunks - their names are appended to the subsection title.
    
    Chunk names are formatted as:
    1. "section_name" - if section has no subsections
    2. "section_name: subsection_name" - if subsection has no subsubsections
    3. "section_name: subsection_name: subsubsection1, subsubsection2, ..." - if subsubsections exist
    """
    
    # Pattern for sections: ALL CAPS title (e.g., "3 METHODS\n")
    section_pattern = r"(?:^|\n)(?P<number>\d+)\s+(?P<title>[A-Z][A-Z0-9\s\-:()]+)(?=\n)"
    
    # Pattern for subsections: Dotted number with one dot (e.g., "3.1 Data collection\n")
    subsection_pattern = r"(?:^|\n)(?P<number>\d+\.\d+)\s+(?P<title>[A-Z][A-Za-z0-9\s\-:()]+)(?=\n)"
    
    # Pattern for subsubsections: Dotted number with two or more dots (e.g., "3.1.1 Details\n")
    subsubsection_pattern = r"(?:^|\n)(?P<number>\d+\.\d+\.\d+(?:\.\d+)*)\s+(?P<title>[A-Z][A-Za-z0-9\s\-:()]+)(?=\n)"
    
    # Find all matches with their details
    matches = []
    
    # Find sections
    for match in re.finditer(section_pattern, text):
        number = match.group("number")
        title = match.group("title").strip()
        alpha_chars = ''.join(c for c in title if c.isalpha())
        if alpha_chars and alpha_chars.isupper():
            matches.append({
                'number': number,
                'title': title,
                'start': match.start(),
                'end_header': match.end(),
                'type': 'section'
            })
    
    # Find subsections
    for match in re.finditer(subsection_pattern, text):
        number = match.group("number")
        title = match.group("title").strip()
        matches.append({
            'number': number,
            'title': title,
            'start': match.start(),
            'end_header': match.end(),
            'type': 'subsection'
        })
    
    # Find subsubsections (for name extraction only)
    subsubsections_by_subsection = {}
    for match in re.finditer(subsubsection_pattern, text):
        number = match.group("number")
        title = match.group("title").strip()
        parent_subsection = '.'.join(number.split('.')[:2])
        if parent_subsection not in subsubsections_by_subsection:
            subsubsections_by_subsection[parent_subsection] = []
        subsubsections_by_subsection[parent_subsection].append(title)
    
    matches = sorted(matches, key=lambda x: x['start'])
    
    sections = {}
    
    for i, match in enumerate(matches):
        start = match['start']
        end = matches[i+1]['start'] if i+1 < len(matches) else len(text)
        content = text[start:end].strip()
        
        number_parts = match['number'].split('.')
        
        if len(number_parts) == 1:
            # This is a main section
            has_subsections = False
            if i+1 < len(matches):
                next_number = matches[i+1]['number']
                if next_number.startswith(match['number'] + '.'):
                    has_subsections = True
            
            if not has_subsections:
                # Section without subsections
                sections[match['title']] = content
        
        else:
            # This is a subsection (e.g., "3.1")
            parent_section_number = number_parts[0]
            
            # Find the parent section
            parent_match = None
            parent_content_before_subsection = ""
            
            for j in range(i-1, -1, -1):
                if matches[j]['number'] == parent_section_number:
                    parent_match = matches[j]
                    parent_start = parent_match['start']
                    parent_text = text[parent_start:start].strip()
                    parent_lines = parent_text.split('\n', 1)
                    if len(parent_lines) > 1:
                        parent_content_before_subsection = parent_lines[1].strip()
                    break
            
            # Build chunk name
            chunk_name = match['title']
            if parent_match:
                chunk_name = f"{parent_match['title']}: {match['title']}"
            
            # Check if this subsection has subsubsections and append their names
            subsection_number = match['number']
            if subsection_number in subsubsections_by_subsection:
                subsubsection_names = ", ".join(subsubsections_by_subsection[subsection_number])
                chunk_name = f"{chunk_name}: {subsubsection_names}"
            
            # Combine parent section info with subsection content
            if parent_match:
                combined_content = f"{parent_content_before_subsection}\n\n{content}"
            else:
                combined_content = content
            
            sections[chunk_name] = combined_content
    
    return sections