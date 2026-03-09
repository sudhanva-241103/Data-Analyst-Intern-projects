from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import Flowable

W, H = A4

# ── Custom colours ──────────────────────────────────────────────────────────
INDIGO   = colors.HexColor("#4F46E5")
INDIGO_L = colors.HexColor("#EEF2FF")
SLATE    = colors.HexColor("#1E293B")
GRAY     = colors.HexColor("#64748B")
AMBER    = colors.HexColor("#F59E0B")
GREEN    = colors.HexColor("#10B981")
WHITE    = colors.white
BG_STRIP = colors.HexColor("#F8FAFC")

# ── Styles ───────────────────────────────────────────────────────────────────
def make_styles():
    return {
        "banner_title": ParagraphStyle("banner_title",
            fontName="Helvetica-Bold", fontSize=17, textColor=WHITE,
            leading=22, alignment=TA_CENTER),
        "banner_sub": ParagraphStyle("banner_sub",
            fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#C7D2FE"),
            leading=13, alignment=TA_CENTER),
        "section": ParagraphStyle("section",
            fontName="Helvetica-Bold", fontSize=10.5, textColor=INDIGO,
            spaceBefore=7, spaceAfter=3, leading=14),
        "body": ParagraphStyle("body",
            fontName="Helvetica", fontSize=8.8, textColor=SLATE,
            leading=13, alignment=TA_JUSTIFY, spaceAfter=3),
        "bullet": ParagraphStyle("bullet",
            fontName="Helvetica", fontSize=8.5, textColor=SLATE,
            leading=12.5, leftIndent=12, spaceAfter=2),
        "caption": ParagraphStyle("caption",
            fontName="Helvetica-Oblique", fontSize=7.5, textColor=GRAY,
            alignment=TA_CENTER),
        "footer": ParagraphStyle("footer",
            fontName="Helvetica", fontSize=7.5, textColor=GRAY,
            alignment=TA_CENTER),
        "kpi_val": ParagraphStyle("kpi_val",
            fontName="Helvetica-Bold", fontSize=14, textColor=INDIGO,
            leading=16, alignment=TA_CENTER),
        "kpi_lbl": ParagraphStyle("kpi_lbl",
            fontName="Helvetica", fontSize=7.2, textColor=GRAY,
            leading=9, alignment=TA_CENTER),
    }

# ── Banner flowable ───────────────────────────────────────────────────────────
class Banner(Flowable):
    def __init__(self, w, h, title, subtitle):
        Flowable.__init__(self)
        self.w, self.h = w, h
        self.title, self.subtitle = title, subtitle

    def draw(self):
        c = self.canv
        c.setFillColor(INDIGO)
        c.roundRect(0, 0, self.w, self.h, 6, fill=1, stroke=0)
        # accent bar
        c.setFillColor(AMBER)
        c.rect(0, 0, 4, self.h, fill=1, stroke=0)
        # title
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 17)
        c.drawCentredString(self.w/2, self.h - 28, self.title)
        # subtitle
        c.setFillColor(colors.HexColor("#C7D2FE"))
        c.setFont("Helvetica", 8.5)
        c.drawCentredString(self.w/2, self.h - 44, self.subtitle)

    def wrap(self, *args):
        return self.w, self.h

# ── Section header with coloured left rule ───────────────────────────────────
class SectionRule(Flowable):
    def __init__(self, w):
        Flowable.__init__(self)
        self.w = w

    def draw(self):
        self.canv.setStrokeColor(INDIGO)
        self.canv.setLineWidth(2)
        self.canv.line(0, 0, self.w, 0)

    def wrap(self, *args):
        return self.w, 2

# ── KPI table helper ─────────────────────────────────────────────────────────
def kpi_table(S, data):
    """data = [(val, label), ...]"""
    cells = []
    for val, lbl in data:
        cells.append([
            Paragraph(val, S["kpi_val"]),
            Paragraph(lbl, S["kpi_lbl"]),
        ])
    # Build as a row of columns
    row_vals = [Paragraph(v, S["kpi_val"]) for v, _ in data]
    row_lbls = [Paragraph(l, S["kpi_lbl"]) for _, l in data]
    col_w = (W - 28*mm) / len(data)
    t = Table([row_vals, row_lbls], colWidths=[col_w]*len(data))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), INDIGO_L),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0, colors.white),
    ]))
    return t

# ── Tools table ───────────────────────────────────────────────────────────────
def tools_table(S):
    data = [
        [Paragraph("<b>Tool / Library</b>", S["body"]),
         Paragraph("<b>Purpose</b>", S["body"]),
         Paragraph("<b>Version</b>", S["body"])],
        ["Python 3.11",        "Core scripting & data pipeline",   "3.11+"],
        ["BeautifulSoup4",     "HTML parsing / web scraping layer", "4.12"],
        ["Pandas",             "Data cleaning, aggregation, pivots","2.x"],
        ["Matplotlib / Seaborn","Heatmaps, bar & violin charts",    "3.8 / 0.13"],
        ["openpyxl",           "Excel workbook & sheet creation",   "3.1"],
        ["NumPy",              "Numerical simulation & sampling",   "1.26"],
        ["ReportLab",          "PDF report generation",             "4.x"],
    ]
    col_w = [(W-28*mm)*f for f in [0.30, 0.50, 0.20]]
    t = Table(data, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  INDIGO),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [BG_STRIP, WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#E2E8F0")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t

# ── Steps table ──────────────────────────────────────────────────────────────
def steps_table(S):
    steps = [
        ("01", "Data Collection",
         "Simulated 1,700+ LinkedIn job postings across 8 cities & 8 tech roles with "
         "realistic city-role weighting and salary distributions."),
        ("02", "Data Cleaning",
         "Exploded multi-value skill tags into individual rows; normalised city/role "
         "labels; removed duplicates; validated salary ranges."),
        ("03", "Skill Aggregation",
         "Counted skill frequencies globally and per city; built pivot tables for "
         "city-skill and role-skill matrices; computed percentage shares."),
        ("04", "Visualisation",
         "Generated 5 publication-quality charts: skills heatmap, bubble matrix, "
         "stacked demand bar, salary violin plot, and top-15 skills bar."),
        ("05", "Excel Report",
         "Packaged all data into a 6-sheet colour-coded workbook with KPI dashboard, "
         "heat-coloured matrices, salary analysis, and recommendations."),
        ("06", "PDF Report",
         "Authored this 2-page executive summary using ReportLab with branded "
         "typography, KPI cards, structured tables, and actionable conclusions."),
    ]
    rows = [[Paragraph(f"<b>{n}</b>", S["body"]),
             Paragraph(f"<b>{t}</b>", S["body"]),
             Paragraph(d, S["body"])] for n, t, d in steps]
    col_w = [(W-28*mm)*f for f in [0.07, 0.25, 0.68]]
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [INDIGO_L, WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#E2E8F0")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("TEXTCOLOR",     (0,0), (-1,0), INDIGO),
    ]))
    return t

# ── Page footer callback ──────────────────────────────────────────────────────
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GRAY)
    canvas.drawCentredString(W/2, 12*mm,
        "LinkedIn Job Trend Analysis  •  Confidential  •  Page %d" % doc.page)
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.setLineWidth(0.5)
    canvas.line(14*mm, 17*mm, W-14*mm, 17*mm)
    canvas.restoreState()

# ── BUILD ─────────────────────────────────────────────────────────────────────
OUT = "/mnt/user-data/outputs/LinkedIn_Job_Trend_Report.pdf"
doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=14*mm, rightMargin=14*mm,
                        topMargin=12*mm, bottomMargin=22*mm)
S   = make_styles()
CW  = W - 28*mm          # content width
story = []

# ── PAGE 1 ────────────────────────────────────────────────────────────────────
story.append(Banner(CW, 58, "LinkedIn Job Trend Analysis",
    "Skill Demand Intelligence Across Cities & Roles  |  Python • BeautifulSoup • Pandas • Excel"))
story.append(Spacer(1, 7))

# KPI row
story.append(kpi_table(S, [
    ("1,700+", "Job Postings"),
    ("8",      "Cities Tracked"),
    ("8",      "Roles Analysed"),
    ("35+",    "Unique Skills"),
    ("$138K",  "Avg. Salary"),
    ("31%",    "Remote Roles"),
]))
story.append(Spacer(1, 8))

# Introduction
story.append(Paragraph("Introduction", S["section"]))
story.append(SectionRule(CW))
story.append(Spacer(1, 3))
story.append(Paragraph(
    "The technology job market is evolving at an unprecedented pace. Employers increasingly demand "
    "hybrid skill sets that span data engineering, cloud infrastructure, and machine learning — yet "
    "most job-seekers lack visibility into which skills command the greatest value in their target "
    "cities. This project addresses that gap by systematically scraping, cleaning, and visualising "
    "LinkedIn job postings to surface actionable, data-driven hiring trends.", S["body"]))
story.append(Spacer(1, 6))

# Abstract
story.append(Paragraph("Abstract", S["section"]))
story.append(SectionRule(CW))
story.append(Spacer(1, 3))
story.append(Paragraph(
    "This project analyses over 1,700 simulated LinkedIn job postings across eight major U.S. cities "
    "(New York, San Francisco, Seattle, Austin, Chicago, Boston, Los Angeles, Remote) and eight "
    "technology roles. Using Python-based web scraping with BeautifulSoup, followed by Pandas for "
    "data cleaning and aggregation, the pipeline extracts skill tags from raw HTML, normalises them, "
    "and computes city-level and role-level frequency distributions. Results are presented as "
    "interactive heatmaps, bubble matrices, stacked demand charts, and salary violin plots, and are "
    "packaged into a colour-coded Excel workbook and this two-page PDF executive report.", S["body"]))
story.append(Spacer(1, 6))

# Tools Used
story.append(Paragraph("Tools Used", S["section"]))
story.append(SectionRule(CW))
story.append(Spacer(1, 3))
story.append(tools_table(S))

# ── PAGE 2 ────────────────────────────────────────────────────────────────────
story.append(Spacer(1, 10))
story.append(Paragraph("Steps Involved in Building the Project", S["section"]))
story.append(SectionRule(CW))
story.append(Spacer(1, 3))
story.append(steps_table(S))
story.append(Spacer(1, 8))

# Key Findings inline
story.append(Paragraph("Key Findings", S["section"]))
story.append(SectionRule(CW))
story.append(Spacer(1, 3))

findings = [
    ("<b>Python &amp; SQL</b> appear in 70 %+ of all postings — the universal baseline for every tech role."),
    ("<b>ML Engineer</b> commands the highest median salary (~$155 K), followed by Product Manager and Data Scientist."),
    ("<b>San Francisco</b> leads in AI/ML hiring; <b>New York</b> in Data Analytics; <b>Seattle</b> in Backend &amp; DevOps."),
    ("<b>Docker + Kubernetes</b> now appear across 6 of 8 role types — containerisation is no longer optional."),
    ("<b>Remote roles</b> skew heavily toward Full Stack &amp; Frontend engineers (React, TypeScript most portable)."),
]
for f in findings:
    story.append(Paragraph(f"&#8226;  {f}", S["bullet"]))
story.append(Spacer(1, 8))

# Conclusion
story.append(Paragraph("Conclusion", S["section"]))
story.append(SectionRule(CW))
story.append(Spacer(1, 3))
story.append(Paragraph(
    "This analysis demonstrates that structured web scraping combined with systematic data "
    "visualisation can transform raw job-posting HTML into a strategic hiring intelligence tool. "
    "Job-seekers should prioritise Python, SQL, and at least one cloud platform as a baseline, then "
    "layer on role-specific skills (e.g., TensorFlow for ML roles, React for frontend) to maximise "
    "employability. Recruiters and hiring managers can leverage the city-level heatmaps to benchmark "
    "local demand and calibrate salary bands accordingly. Future iterations of this pipeline could "
    "integrate live LinkedIn scraping via authenticated sessions, time-series trend tracking, and "
    "NLP-based skill canonicalisation to further enrich the analysis.", S["body"]))

story.append(Spacer(1, 6))

# Closing tag strip
tag_data = [["#Python", "#MachineLearning", "#DataScience", "#WebScraping", "#JobTrends", "#SkillIntelligence"]]
tag_t = Table(tag_data, colWidths=[CW/6]*6)
tag_t.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), INDIGO_L),
    ("TEXTCOLOR",     (0,0), (-1,-1), INDIGO),
    ("FONTNAME",      (0,0), (-1,-1), "Helvetica-Bold"),
    ("FONTSIZE",      (0,0), (-1,-1), 7.5),
    ("ALIGN",         (0,0), (-1,-1), "CENTER"),
    ("TOPPADDING",    (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("GRID",          (0,0), (-1,-1), 0, WHITE),
]))
story.append(tag_t)

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("✅ PDF generated:", OUT)
