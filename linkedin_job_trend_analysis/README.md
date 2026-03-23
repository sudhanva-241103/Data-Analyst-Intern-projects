# 📊 LinkedIn Job Trend Analysis

> Scrape LinkedIn job postings to analyze skill demand trends across cities and roles — with heatmaps, salary insights, and an Excel + PDF report.

---

## 🗂️ Project Structure

```
linkedin_job_trend_analysis/
│
├── linkedin_analysis.py      # Main pipeline: data simulation, analysis & visualizations
├── generate_report.py        # PDF report generator (ReportLab)
├── requirements.txt          # All Python dependencies
│
├── data/                     # (Auto-generated) raw/cleaned data CSVs
├── outputs/                  # (Auto-generated) charts, Excel workbook, PDF report
│   ├── heatmap_skills_by_city.png
│   ├── skill_role_matrix.png
│   ├── job_demand_by_city.png
│   ├── salary_by_role.png
│   ├── top15_skills.png
│   ├── LinkedIn_Job_Trend_Analysis.xlsx
│   └── LinkedIn_Job_Trend_Report.pdf
│
└── README.md
```

---

## 🎯 Objective

Analyze LinkedIn job postings to surface:
- Which **skills** are most in demand across cities
- Which **roles** pay the most
- City-level **hiring hotspots** for each tech role
- Actionable **career recommendations** for job seekers

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| `Python 3.11` | Core scripting |
| `BeautifulSoup4` | HTML parsing / web scraping layer |
| `Pandas` | Data cleaning, aggregation, pivot tables |
| `Matplotlib` | Charts and visualizations |
| `Seaborn` | Heatmaps and statistical plots |
| `openpyxl` | Excel workbook creation |
| `NumPy` | Numerical simulation and sampling |
| `ReportLab` | PDF report generation |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/linkedin-job-trend-analysis.git
cd linkedin-job-trend-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the analysis

```bash
python linkedin_analysis.py
```

### 4. Generate the PDF report

```bash
python generate_report.py
```

All outputs will be saved in the `outputs/` folder.

---

## 📈 Deliverables

### Visualizations
| Chart | Description |
|-------|-------------|
| `heatmap_skills_by_city.png` | Top skills demand % per city |
| `skill_role_matrix.png` | Bubble matrix of skills × roles |
| `job_demand_by_city.png` | Stacked bar of role demand by city |
| `salary_by_role.png` | Salary distribution violin plot |
| `top15_skills.png` | Top 15 most in-demand skills overall |

### Excel Workbook (`LinkedIn_Job_Trend_Analysis.xlsx`)
- 📊 **Dashboard** — KPI cards + top skills table + recommendations
- 📋 **Job Data** — Full dataset (1,700+ postings)
- 🌡 **Skill×City** — Color-gradient heatmap matrix
- 🎯 **Skill×Role** — Role vs skill count matrix
- 💰 **Salary** — Min/Mean/Median/Max by role
- 💡 **Recommendations** — 12 prioritized career insights

### PDF Report (`LinkedIn_Job_Trend_Report.pdf`)
- 2-page executive summary
- Introduction, Abstract, Tools, Steps, Findings, Conclusion

---

## 🔑 Key Findings

- **Python & SQL** appear in 70%+ of all postings — universal baseline
- **ML Engineer** commands the highest median salary (~$155K)
- **San Francisco** leads in AI/ML hiring; **NYC** in Data Analytics
- **Docker + Kubernetes** appear in 6 of 8 role types
- **Remote roles** skew toward Full Stack & Frontend engineers

---

## ⚠️ Note on Web Scraping

This project currently uses **simulated data** that mirrors real LinkedIn posting patterns. To use live data:
1. Use `BeautifulSoup` + `requests` to scrape LinkedIn search result pages
2. Parse job cards for title, location, and skill tags
3. Respect LinkedIn's `robots.txt` and rate-limit your requests
4. Consider using the [LinkedIn API](https://developer.linkedin.com/) for production use

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

Built as a data analytics portfolio project demonstrating end-to-end web scraping, data visualization, and reporting.
