import re
from typing import Dict

def split_into_sections(text: str) -> Dict[str, str]:
    """
    Split text at the lowest level of headers (leaf nodes).
    
    Each chunk is created at the deepest header level found within a branch.
    Chunk names include the full hierarchical path: "Parent: Grandparent: Node".
    """
    
    # Pattern for all headers: number (any depth) + title
    header_pattern = r"(?:^|\n)(?P<number>\d+(?:\.\d+)*)\s+(?P<title>[A-Z][A-Za-z\s\-:()]+)(?=\n)"
    
    # Find all headers
    matches = []
    for match in re.finditer(header_pattern, text):
        number = match.group("number")
        title = match.group("title").strip()
        
        # Filter: must be valid header (all caps for depth 1, or normal case for deeper)
        number_parts = number.split('.')
        if len(number_parts) == 1:
            # Section header must be all caps
            alpha_chars = ''.join(c for c in title if c.isalpha())
            if not (alpha_chars and alpha_chars.isupper()):
                continue
        
        matches.append({
            'number': number,
            'title': title,
            'depth': len(number.split('.')),
            'start': match.start(),
            'end_header': match.end(),
        })
    
    if not matches:
        return {"full_document": text}
    
    matches = sorted(matches, key=lambda x: x['start'])
    
    # Build parent hierarchy: map number to ancestors
    def get_ancestors(number):
        """Get all ancestor numbers for a given number."""
        parts = number.split('.')
        ancestors = []
        for i in range(1, len(parts)):
            ancestor = '.'.join(parts[:i])
            ancestors.append(ancestor)
        return ancestors
    
    # Find leaf nodes (headers with no children)
    def has_children(match_idx):
        """Check if a header has any children."""
        if match_idx >= len(matches) - 1:
            return False
        
        current_number = matches[match_idx]['number']
        next_number = matches[match_idx + 1]['number']
        
        # Has children if next number starts with current number + "."
        return next_number.startswith(current_number + '.')
    
    sections = {}
    
    for i, match in enumerate(matches):
        # Only create chunks for leaf nodes (no children)
        if has_children(i):
            continue
        
        # Get content from this header to next header (or end of text)
        start = match['start']
        end = matches[i + 1]['start'] if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        
        # Build hierarchical name: include all ancestor titles
        ancestors = get_ancestors(match['number'])
        title_parts = []
        
        for ancestor_num in ancestors:
            for ancestor_match in matches:
                if ancestor_match['number'] == ancestor_num:
                    title_parts.append(ancestor_match['title'])
                    break
        
        # Add current title
        title_parts.append(match['title'])
        chunk_name = ': '.join(title_parts)
        
        sections[chunk_name] = content
    
    return sections