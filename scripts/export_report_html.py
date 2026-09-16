"""
Utility script to convert report.md into an HTML / PDF document.
Includes CSS styling for printing and Mermaid diagram rendering support.
"""

from pathlib import Path
import re


def convert_markdown_to_html(md_path: Path, html_path: Path) -> None:
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Simple HTML generator with modern styling
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionInspect - Project Report</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({startOnLoad:true, theme:'neutral'});</script>
    <style>
        @page {
            size: A4;
            margin: 20mm 15mm 20mm 15mm;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        h1, h2, h3, h4 {
            color: #1a202c;
            font-weight: 600;
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
            margin-top: 1.5em;
        }
        h1 { font-size: 2em; text-align: center; border: none; margin-bottom: 0.5em; }
        h2 { font-size: 1.4em; margin-top: 2em; }
        h3 { font-size: 1.15em; border: none; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5em 0;
            font-size: 0.95em;
        }
        th, td {
            border: 1px solid #d1d5db;
            padding: 8px 14px;
            text-align: left;
        }
        th {
            background-color: #f3f4f6;
            font-weight: 600;
        }
        tr:nth-child(even) {
            background-color: #f9fafb;
        }
        pre {
            background-color: #f6f8fa;
            border-radius: 6px;
            padding: 14px;
            overflow-x: auto;
            font-size: 0.85em;
            line-height: 1.45;
            border: 1px solid #e1e4e8;
        }
        code {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            background-color: rgba(27,31,35,0.05);
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 85%;
        }
        pre code {
            background: none;
            padding: 0;
        }
        .mermaid {
            display: flex;
            justify-content: center;
            margin: 25px 0;
            background: #ffffff;
            padding: 15px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
        }
        @media print {
            body { padding: 0; }
            h2 { page-break-before: always; }
        }
    </style>
</head>
<body>
<!-- CONTENT_PLACEHOLDER -->
</body>
</html>
"""

    # Convert code fences and mermaid blocks
    processed = []
    in_code = False
    in_mermaid = False
    code_buffer = []

    for line in md_text.splitlines():
        if line.startswith("```mermaid"):
            in_mermaid = True
            code_buffer = []
            continue
        elif in_mermaid and line.startswith("```"):
            in_mermaid = False
            processed.append(f'<div class="mermaid">\n' + "\n".join(code_buffer) + "\n</div>")
            code_buffer = []
            continue
        elif in_mermaid:
            code_buffer.append(line)
            continue

        if line.startswith("```"):
            if in_code:
                in_code = False
                processed.append("<pre><code>" + "\n".join(code_buffer) + "</code></pre>")
                code_buffer = []
            else:
                in_code = True
                code_buffer = []
            continue
        elif in_code:
            code_buffer.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Headers
        if line.startswith("# "):
            processed.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            processed.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            processed.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("#### "):
            processed.append(f"<h4>{line[5:]}</h4>")
        elif line.startswith("|"):
            # Render markdown table row
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c).issubset({"-", ":", " "}) for c in cells):
                continue  # divider row
            row_html = "".join(f"<td>{c}</td>" for c in cells)
            processed.append(f"<tr>{row_html}</tr>")
        elif line.strip() == "---":
            processed.append("<hr/>")
        elif line.strip():
            # Paragraph or list
            if line.startswith("- "):
                processed.append(f"<li>{line[2:]}</li>")
            else:
                processed.append(f"<p>{line}</p>")

    body_content = "\n".join(processed)
    # Wrap table rows
    body_content = re.sub(r'(<tr>.*?</tr>\s*)+', r'<table>\g<0></table>', body_content, flags=re.DOTALL)

    final_html = html_template.replace("<!-- CONTENT_PLACEHOLDER -->", body_content)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"[+] Exported report HTML: {html_path}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    md = root / "report.md"
    html = root / "report.html"
    convert_markdown_to_html(md, html)
