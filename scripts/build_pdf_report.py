"""
Direct PDF Generator for VisionInspect Academic Project Report.
Uses ReportLab to build a professional, styled, publication-ready PDF document.
"""

from pathlib import Path
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds professional running header and footer with dynamic page counts."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4b5563"))

        # Don't draw header/footer on cover page (page 1)
        if self._pageNumber > 1:
            # Header
            self.drawString(54, 750, "VisionInspect: Industrial Surface Defect Detection & Quality Inspection System")
            self.drawRightString(558, 750, "CSE3010 - Computer Vision Project")
            self.setStrokeColor(colors.HexColor("#e5e7eb"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

            # Footer
            self.line(54, 45, 558, 45)
            self.drawString(54, 32, "VITyarthi Flipped Course Evaluation | Student: Manish Kumar Rathore (24BAI10931)")
            self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")

        self.restoreState()


def build_pdf(md_path: Path, output_pdf_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1e3a8a"),
        alignment=1, # Center
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1f2937"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#374151"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#111827"),
        backColor=colors.HexColor("#f3f4f6"),
        borderColor=colors.HexColor("#e5e7eb"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8
    )

    table_header_style = ParagraphStyle(
        'THStyle',
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TDStyle',
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1f2937")
    )

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    story = []
    lines = md_text.splitlines()
    i = 0
    in_code_block = False
    code_lines = []

    while i < len(lines):
        line = lines[i]

        # Handle Code Fences
        if line.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "<br/>".join(code_lines).replace(" ", "&nbsp;")
                story.append(Paragraph(code_text, code_style))
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            # Skip raw mermaid diagrams in PDF flow and replace with compact note
            if any(k in line for k in ["classDiagram", "erDiagram", "sequenceDiagram", "flowchart"]):
                code_lines.append(f"<i>[Diagram: {line.strip()}]</i>")
            else:
                clean_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                code_lines.append(clean_line)
            i += 1
            continue

        # Markdown Headers
        if line.startswith("# "):
            story.append(Paragraph(line[2:], title_style))
            story.append(Spacer(1, 10))
        elif line.startswith("## "):
            sec_title = line[3:]
            # If start of major section after cover page, page break
            if sec_title.startswith("2. Introduction"):
                story.append(PageBreak())
            story.append(Paragraph(sec_title, h1_style))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], h2_style))
        elif line.startswith("#### "):
            story.append(Paragraph(line[5:], h2_style))
        elif line.startswith("|"):
            # Collect entire markdown table
            table_rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row_str = lines[i].strip().strip("|")
                cells = [c.strip() for c in row_str.split("|")]
                if not all(set(c).issubset({"-", ":", " "}) for c in cells):
                    table_rows.append(cells)
                i += 1
            i -= 1  # Adjust index

            if table_rows:
                # Build reportlab table
                flowable_data = []
                for row_idx, row in enumerate(table_rows):
                    row_data = []
                    for cell in row:
                        cell_clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', cell)
                        cell_clean = re.sub(r'\*(.*?)\*', r'<i>\1</i>', cell_clean)
                        cell_clean = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', cell_clean)
                        if row_idx == 0:
                            row_data.append(Paragraph(cell_clean, table_header_style))
                        else:
                            row_data.append(Paragraph(cell_clean, table_cell_style))
                    flowable_data.append(row_data)

                col_count = len(flowable_data[0])
                total_w = 504  # 612 - 108
                col_w = total_w / col_count

                t = Table(flowable_data, colWidths=[col_w] * col_count)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ]))
                story.append(t)
                story.append(Spacer(1, 8))
        elif line.startswith("- "):
            text = line[2:]
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text)
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{text}", bullet_style))
        elif line.strip() == "---":
            story.append(Spacer(1, 4))
        elif line.strip():
            text = line
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text)
            story.append(Paragraph(text, body_style))

        i += 1

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] Successfully generated academic PDF: {output_pdf_path}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    md_file = root / "report.md"
    pdf_file = root / "VisionInspect_Project_Report.pdf"
    build_pdf(md_file, pdf_file)
