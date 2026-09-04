#!/usr/bin/env python3
"""Compiles presentation/presentation.md to a publication-quality presentation PDF with vector SVG math."""

import html
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import markdown
import weasyprint

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

PRESENTATION_DIR = Path("presentation")
MD_FILE = PRESENTATION_DIR / "presentation.md"
PDF_FILE = PRESENTATION_DIR / "presentation.pdf"

MATH_CACHE = {}


def render_latex_to_svg(latex_code: str, fontsize: int = 13) -> str:
    key = (latex_code, fontsize)
    if key in MATH_CACHE:
        return MATH_CACHE[key]

    clean_tex = latex_code.strip()
    clean_tex = clean_tex.replace(r"\text{", r"\mathrm{")
    clean_tex = clean_tex.replace(r"\textbf{", r"\mathbf{")
    clean_tex = clean_tex.replace(r"\le", r"\le")
    clean_tex = clean_tex.replace(r"\ge", r"\ge")
    clean_tex = clean_tex.replace(r"\approx", r"\approx")

    fig = plt.figure(figsize=(0.01, 0.01))
    try:
        fig.text(0, 0, f"${clean_tex}$", fontsize=fontsize)
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.04, transparent=True)
        svg_code = buf.getvalue()
        if "<?xml" in svg_code:
            svg_code = svg_code[svg_code.find("<svg") :]
        res = f'<div class="display-math">{svg_code}</div>'
        MATH_CACHE[key] = res
        return res
    except Exception as e:
        escaped = html.escape(latex_code)
        res = f'<div class="display-math"><code>{escaped}</code></div>'
        MATH_CACHE[key] = res
        return res
    finally:
        plt.close(fig)


CSS_STYLES = """
@page {
    size: A4 portrait;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-left {
        content: "Discrete Flow Matching for De Novo Peptide Sequencing | Joel Gedeon (AIMS)";
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 8pt;
        color: #718096;
    }
    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 8pt;
        font-weight: 600;
        color: #4a5568;
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #2d3748;
    line-height: 1.55;
    font-size: 10pt;
}

h1 {
    color: #1a365d;
    font-size: 18pt;
    font-weight: 800;
    margin-top: 0;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}

h2 {
    color: #2b6cb0;
    font-size: 13pt;
    font-weight: 700;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 4px;
    margin-top: 24px;
    margin-bottom: 10px;
    page-break-after: avoid;
}

/* Page break before key presentation sections */
h2#task-understanding-de-novo-peptide-sequencing,
h2#3-model-architecture-components,
h2#6-experimental-results-progression,
h2#7-supervisor-qa-anticipated-questions-technical-answers,
h2#8-technical-notes-for-supervisor-score-calibration-quality-metrics {
    page-break-before: always;
}

h3 {
    color: #2d3748;
    font-size: 10.8pt;
    font-weight: 700;
    margin-top: 14px;
    margin-bottom: 5px;
    page-break-after: avoid;
}

h4 {
    color: #4a5568;
    font-size: 9.8pt;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
}

p {
    margin-top: 4px;
    margin-bottom: 8px;
}

ul, ol {
    margin-top: 4px;
    margin-bottom: 10px;
    padding-left: 22px;
}

li {
    margin-bottom: 4px;
}

/* Presentation title block */
.title-card {
    background: linear-gradient(135deg, #ebf8ff 0%, #edf2f7 100%);
    border-left: 6px solid #3182ce;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 18px;
}

.title-card h1 {
    margin-bottom: 4px;
}

.title-card .sub {
    font-size: 11.5pt;
    color: #4a5568;
    font-weight: 600;
    margin-bottom: 10px;
}

.title-card .meta {
    font-size: 8.8pt;
    color: #4a5568;
    line-height: 1.45;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 16px 0;
    font-size: 9pt;
    page-break-inside: avoid;
}

th, td {
    padding: 7px 10px;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
}

th {
    background-color: #2b6cb0;
    color: #ffffff;
    font-weight: 700;
    font-size: 8.5pt;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

tr:nth-child(even) td {
    background-color: #f7fafc;
}

/* Code & ASCII Diagrams */
pre {
    background-color: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 14px;
    font-family: "JetBrains Mono", Consolas, Menlo, monospace;
    font-size: 8pt;
    line-height: 1.4;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 10px 0;
}

code {
    background-color: #edf2f7;
    color: #805ad5;
    padding: 1px 4px;
    border-radius: 4px;
    font-family: Consolas, monospace;
    font-size: 8.5pt;
}

pre code {
    background-color: transparent;
    color: #2d3748;
    padding: 0;
}

/* Callout blockquotes for Q&A and notes */
blockquote {
    border-left: 4px solid #319795;
    background-color: #e6fffa;
    margin: 8px 0 14px 0;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 9.3pt;
    page-break-inside: avoid;
}

blockquote p {
    margin: 3px 0;
}

/* Images & Figures */
img {
    max-width: 95%;
    max-height: 420px;
    height: auto;
    display: block;
    margin: 12px auto;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    page-break-inside: avoid;
}

/* Math blocks */
.display-math {
    margin: 10px 0;
    text-align: center;
    page-break-inside: avoid;
}

.display-math svg {
    max-width: 90%;
    height: auto;
    display: inline-block;
}

hr {
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 18px 0;
}
"""


def main():
    if not MD_FILE.exists():
        raise FileNotFoundError(f"Missing {MD_FILE}")

    raw_text = MD_FILE.read_text(encoding="utf-8")

    # 1. Extract display math $$...$$ and replace with unique placeholders
    math_placeholders = {}

    def stash_math(match):
        idx = len(math_placeholders)
        placeholder = f"MATHPLACEHOLDER{idx}END"
        math_placeholders[placeholder] = match.group(1).strip()
        return f"\n\n{placeholder}\n\n"

    text_with_placeholders = re.sub(r"\$\$(.+?)\$\$", stash_math, raw_text, flags=re.DOTALL)

    # 2. Convert Markdown to HTML
    print("Converting Markdown to HTML...")
    md_converter = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "attr_list", "def_list"]
    )
    html_body = md_converter.convert(text_with_placeholders)

    # 3. Replace math placeholders with rendered vector SVGs
    print(f"Rendering {len(math_placeholders)} display equations to vector SVGs...")
    for placeholder, latex_code in math_placeholders.items():
        svg_div = render_latex_to_svg(latex_code, fontsize=13)
        # In HTML, placeholder might be wrapped in <p>...</p>
        html_body = html_body.replace(f"<p>{placeholder}</p>", svg_div)
        html_body = html_body.replace(placeholder, svg_div)

    # 4. Wrap presentation title block in a styled title card
    html_body = re.sub(
        r"<h1>(.*?)</h1>\s*<h2>(.*?)</h2>\s*<p><strong>Presenter:</strong> (.*?)<br />\s*<strong>Affiliation:</strong> (.*?)<br />\s*<strong>Focus:</strong> (.*?)<br />\s*<strong>Date:</strong> (.*?)</p>",
        r'<div class="title-card"><h1>\1</h1><div class="sub">\2</div><div class="meta"><strong>Presenter:</strong> \3 | <strong>Affiliation:</strong> \4<br/><strong>Focus:</strong> \5 | <strong>Date:</strong> \6</div></div>',
        html_body,
        flags=re.DOTALL,
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DFM De Novo Peptide Sequencing Presentation</title>
<style>
{CSS_STYLES}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    html_path = PRESENTATION_DIR / "presentation.html"
    html_path.write_text(full_html, encoding="utf-8")
    print(f"Generated HTML at {html_path}")

    print(f"Compiling PDF with WeasyPrint to {PDF_FILE}...")
    doc = weasyprint.HTML(filename=str(html_path), base_url=str(PRESENTATION_DIR))
    doc.write_pdf(target=str(PDF_FILE))
    print(f"Successfully compiled {PDF_FILE} ({PDF_FILE.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
