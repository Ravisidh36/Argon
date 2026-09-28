"""
Quick debug: dump the raw markdown that MarkItDown produced for a file,
BEFORE any cleaning or chunking. Use this to check whether real markdown
headers (#, ##, ###) actually exist in the output.

Usage:
    python -m eval.dump_markdown data/documents/Articles.pdf
"""

import sys
from tools.document_understanding import doc_reader


def dump(path):
    markdown = doc_reader(path)
    print(markdown)
    print("\n\n--- HEADER LINES FOUND ---")
    header_lines = [line for line in markdown.split("\n") if line.strip().startswith("#")]
    if not header_lines:
        print("(none)")
    else:
        for line in header_lines:
            print(line)


if __name__ == "__main__":
    path = sys.argv[1]
    dump(path)
