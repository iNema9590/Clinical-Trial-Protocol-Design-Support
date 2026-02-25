import pdfplumber
import pandas as pd
import re
from pdfplumber.utils import extract_text, get_bbox_overlap, obj_to_bbox
from typing import List, Optional, Counter

# remove the headnotes (first two lines) and footnotes (last line) from the text
def remove_notes(text):
    lines = text.split("\n")
    if len(lines) > 3:
        return "\n".join(lines[2:-1])
    else:
        return text    


def process_pdf(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    all_text = []
    for page in pdf.pages[11:110]: #  skip everything before ToC and after Appendix
        filtered_page = page
        chars = filtered_page.chars
        for table in page.find_tables():
            first_table_char = page.crop(table.bbox).chars[0]
            filtered_page = filtered_page.filter(lambda obj: 
                get_bbox_overlap(obj_to_bbox(obj), table.bbox) is None
            )
            chars = filtered_page.chars
            df = pd.DataFrame(table.extract())
            df.columns = df.iloc[0]
            markdown = df.drop(0).to_markdown(index=False)
            chars.append(first_table_char | {"text": markdown})
        page_text = extract_text(chars, layout=True)
        page_text = remove_notes(page_text)
        # remove multiple newlines
        page_text = re.sub(r'\n+', '\n', page_text)
        all_text.append(page_text)
    pdf.close()
    return "\n".join(all_text)


