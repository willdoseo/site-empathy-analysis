# Site Empathy Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Analyze website content for empathy using NLP** — powered by [Firecrawl](https://firecrawl.dev) and the Sharma et al. (2020) empathy framework.

<p align="center">
  <img src="https://via.placeholder.com/600x300?text=Site+Empathy+Analysis" alt="Site Empathy Analysis" width="600">
</p>

## 🎯 What It Does

Site Empathy Analysis crawls websites and measures how empathetic their content is across three dimensions:

| Dimension | Description | Example |
|-----------|-------------|---------|
| 🔥 **Emotional Reactions** | Warmth, compassion, concern | "I'm so sorry you're going through this" |
| 🧠 **Interpretations** | Understanding, acknowledging feelings | "It sounds like you're feeling overwhelmed" |
| 💬 **Explorations** | Engagement, inviting dialogue | "Can you tell me more about what happened?" |

The tool provides **comprehensive reporting** with:
- 📊 **Interactive HTML dashboards** with charts and visualizations
- 📈 **Enhanced terminal output** with histograms and progress bars
- 📁 **Multiple export formats** (CSV, JSON, HTML)
- 🎯 **Actionable insights** with specific recommendations
- 📑 **Page-level grades** (A-F) and empathy profiles
- 🔝 **Top/bottom page identification** with content previews

## 🚀 Quick Start

### Installation

```bash
pip install git+https://github.com/willdoseo/site-empathy-analysis.git
```

### Get Your API Key

1. Sign up at [firecrawl.dev](https://firecrawl.dev)
2. Get your API key
3. Set it as an environment variable:

```bash
export FIRECRAWL_API_KEY="fc-your-api-key"
```

Or configure it interactively:

```bash
site-empathy config --set-key fc-your-api-key
```

### Analyze a Website

```bash
# Analyze a single site
site-empathy analyze https://example.com -o report.csv

# Analyze with custom page limit
site-empathy analyze example.com --max-pages 50 -o report.csv

# Analyze multiple sites
site-empathy batch sites.txt -o reports/
```

### ✨ What You'll See

The tool provides **rich terminal output** with:
- 📊 Overall grade (A-F) and key metrics
- 📈 Visual progress bars for each empathy dimension
- 📉 ASCII histogram of score distribution
- 🌟 Top 5 performing pages with grades and profiles
- ⚠️ Bottom 5 pages needing attention
- 💡 Actionable insights and specific recommendations
- 📁 Export suggestions for HTML, CSV, and JSON formats

**Plus beautiful HTML reports** with:
- 📊 Interactive Chart.js visualizations
- 🎯 Detailed page-by-page breakdown with content previews
- 🌈 Color-coded grades and empathy profiles
- 💬 Language analysis with most common phrases
- 📈 Score distribution charts and dimension radar plots

## 📊 Output Format

### Page-Level CSV

| Column | Description |
|--------|-------------|
| `url` | Page URL |
| `domain` | Site domain |
| `title` | Page title |
| `empathy_score` | Combined empathy score (0-1) |
| `emotional_reaction_score` | ER dimension score |
| `interpretation_score` | IP dimension score |
| `exploration_score` | EX dimension score |
| `er_level` | ER level (0=none, 1=weak, 2=strong) |
| `ip_level` | IP level |
| `ex_level` | EX level |
| `empathic_phrases` | Detected empathic language |
| `non_empathic_indicators` | Clinical/transactional language |

### Example Output

```
url,domain,title,empathy_score,emotional_reaction_score,...
https://example.com/about,example.com,About Us,0.342,0.28,...
https://example.com/services,example.com,Our Services,0.156,0.05,...
```

## 💻 Python API

### Basic Usage

```python
from site_empathy_analysis import SiteEmpathyAnalyzer

# Initialize
analyzer = SiteEmpathyAnalyzer(firecrawl_key="fc-your-key")

# Analyze a site
result = analyzer.analyze_site("https://example.com", max_pages=100)

# Access results
print(f"Mean empathy: {result.mean_empathy_score:.3f}")
print(f"Pages with emotional warmth: {result.pct_with_emotional_reaction:.1f}%")

# Get overall grade (A-F)
print(f"Overall grade: {result._get_overall_grade()}")
```

### 📊 Export Options

```python
# 1. HTML Report (Interactive Dashboard with Charts)
result.to_html("empathy_report.html")
# → Beautiful report with Chart.js visualizations, top/bottom pages, insights

# 2. CSV Export (Page-level data for Excel/Tableau)
result.to_csv("empathy_data.csv")
# → Spreadsheet-ready format with all metrics

# 3. JSON Export (Complete analysis data)
result.to_json("empathy_full.json")
# → Structured data for APIs, custom dashboards, further processing

# 4. DataFrame (Python analysis)
df = result.to_dataframe()
# → Pandas DataFrame for data science workflows
```

### 📈 Advanced Analysis

```python
# Get score distribution
distribution = result.get_score_distribution()
print(distribution)  # {'F (0-0.1)': 5, 'D (0.1-0.2)': 3, ...}

# Get top performing pages
top_pages = result.get_top_pages(5)
for page in top_pages:
    print(f"{page.title}: {page.empathy_score:.3f} ({page.get_grade()})")
    print(f"  Profile: {page.get_empathy_profile()}")
    print(f"  Empathic phrases: {page.empathic_phrases[:3]}")

# Get pages needing improvement
bottom_pages = result.get_bottom_pages(5)
for page in bottom_pages:
    print(f"⚠️ {page.title}: {page.empathy_score:.3f}")
    if page.er_level == 0:
        print("  → Add emotional warmth")
    if page.ip_level == 0:
        print("  → Show more understanding")

# Filter DataFrame by criteria
high_empathy = df[df['empathy_score'] >= 0.3]
print(f"High empathy pages: {len(high_empathy)}/{len(df)}")

# Find pages with specific patterns
warmth_no_understanding = [
    p for p in result.pages 
    if p.er_level >= 1 and p.ip_level == 0
]
```

### Batch Analysis

```python
urls = [
    "https://site1.com",
    "https://site2.com",
    "https://site3.com",
]

results = analyzer.analyze_batch(
    urls,
    max_pages_per_site=50,
    output_dir="reports/",
)
```

### Quick Text Analysis

```python
from site_empathy_analysis.models import EmpathyScorer

scorer = EmpathyScorer()
result = scorer.score("We understand how difficult this time must be for you.")

print(f"Empathy: {result.empathy_score:.3f}")
print(f"Empathic phrases: {result.empathic_phrases}")
```

## 🛠️ CLI Reference

### `site-empathy analyze`

Analyze a single website.

```bash
site-empathy analyze URL [OPTIONS]

Options:
  -o, --output PATH    Output CSV file path
  --max-pages INT      Maximum pages to analyze (default: 100)
  --quiet              Minimal output
  --key TEXT           Firecrawl API key
```

### `site-empathy batch`

Analyze multiple websites from a file.

```bash
site-empathy batch INPUT_FILE [OPTIONS]

Options:
  -o, --output PATH    Output directory (required)
  --max-pages INT      Max pages per site (default: 50)
  --summary PATH       Path for summary CSV
  --quiet              Minimal output
```

### `site-empathy config`

Configure settings.

```bash
site-empathy config [OPTIONS]

Options:
  --set-key TEXT    Set Firecrawl API key
  --show-key        Show saved API key
  --clear-key       Remove saved API key
```

### `site-empathy quick`

Quick analysis of a text string.

```bash
site-empathy quick "We understand how difficult this is for you."
```

### `site-empathy info`

Show information about the empathy framework.

```bash
site-empathy info
```

## 📈 Understanding Scores

### Letter Grades (New!)

Each page and site receives an overall grade:

| Grade | Score Range | Description |
|-------|-------------|-------------|
| **A** | 0.40+ | Excellent empathy - warm, understanding, engaging |
| **B** | 0.30-0.40 | Good empathy - solid warmth and understanding |
| **C** | 0.20-0.30 | Moderate - some empathy but room for improvement |
| **D** | 0.10-0.20 | Low empathy - mostly clinical/transactional tone |
| **F** | <0.10 | Very low - lacks emotional connection |

### Empathy Profiles (New!)

Pages are categorized by which dimensions they exhibit:

| Profile | Dimensions | Description |
|---------|-----------|-------------|
| 🌟 **Full Empathy** | ER + IP + EX | Warm, understanding, AND engaging |
| 💚 **Empathic** | ER + IP | Warm and understanding (most desirable) |
| 💛 **Warm & Engaging** | ER + EX | Friendly but lacks depth |
| 💙 **Understanding & Engaging** | IP + EX | Informative dialogue but cold |
| 🟠 **Emotionally Warm Only** | ER only | Caring but superficial |
| 🔵 **Understanding Only** | IP only | Clinical - explains but doesn't connect |
| 🟣 **Engaging Only** | EX only | Asks questions without warmth |
| ⚪ **Minimal Empathy** | None | Transactional/factual only |

### Empathy Score Scale

| Score | Level | Description |
|-------|-------|-------------|
| 0.35+ | High | Strong emotional warmth and understanding |
| 0.20-0.35 | Moderate | Some warmth, room for improvement |
| 0.10-0.20 | Low | Mostly clinical/transactional |
| <0.10 | Very Low | Lacks emotional warmth |

### Dimension Levels

Each dimension (ER, IP, EX) is scored 0-2:
- **0** = No expression of this dimension
- **1** = Weak expression
- **2** = Strong expression

### Combined Score Formula

```
empathy_score = (ER × 0.5) + (IP × 0.3) + (EX × 0.2)
```

Where each dimension score is calculated as:
```
dimension_score = (prob_level_1 × 0.5) + (prob_level_2 × 1.0)
```

## 🔑 Empathic Language Patterns

The analyzer detects these empathic phrases:

| Category | Examples |
|----------|----------|
| **Warmth** | "every step", "your journey", "we care", "we understand" |
| **Understanding** | "going through", "you may feel", "many people", "it can be" |
| **Support** | "here for you", "not alone", "by your side", "whenever you're ready" |
| **Encouragement** | "courage", "strength", "brave", "taking this step" |

And flags non-empathic indicators:

| Category | Examples |
|----------|----------|
| **Transactional** | "click here", "submit", "terms and conditions" |
| **Clinical** | "evidence-based", "best practices", "statistics show" |
| **Impersonal** | "founded in", "completion rate", "serving since" |

## 📚 Attribution

The empathy analysis framework is based on:

> **Sharma, A., Miner, A.S., Atkins, D.C., & Althoff, T. (2020).**  
> *A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support.*  
> Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).  
> https://aclanthology.org/2020.emnlp-main.425/

```bibtex
@inproceedings{sharma2020empathy,
    title={A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support},
    author={Sharma, Ashish and Miner, Adam S and Atkins, David C and Althoff, Tim},
    year={2020},
    booktitle={EMNLP}
}
```

## 🧠 Model Setup

The empathy analysis requires trained model weights (~3GB total). On first use, the package will attempt to download them automatically.

### Model Management

```bash
# Check model status
site-empathy models --status

# Download models
site-empathy models --download

# Clear cached models
site-empathy models --clear
```

### Model Locations

Models are searched in this order:
1. `~/.cache/site-empathy/models/` (downloaded cache)
2. Package `models/` directory (local install)

### Self-Hosting Models

For production use or if the default download fails, you can host your own models:

1. **Upload the model files** to your server or cloud storage:
   - `modern_ER.pth` (~960MB) - Emotional Reactions
   - `modern_IP.pth` (~960MB) - Interpretations
   - `modern_EX.pth` (~960MB) - Explorations

2. **Set the environment variable**:
   ```bash
   export SITE_EMPATHY_MODEL_URL="https://your-server.com/models"
   ```

3. Models will be downloaded from `{SITE_EMPATHY_MODEL_URL}/modern_ER.pth`, etc.

### HuggingFace Hub

Models are hosted on HuggingFace and download automatically on first use:

**Models:** [huggingface.co/willdoseo/site-empathy-models](https://huggingface.co/willdoseo/site-empathy-models)

## 🔒 Privacy & Security

- **No data storage**: This tool doesn't store your crawled content beyond local CSV exports
- **API keys**: Never commit API keys to version control
- **Respect robots.txt**: Firecrawl respects website crawling rules

## 📋 Requirements

- Python 3.9+
- Firecrawl API key
- ~2GB RAM for model inference

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ for better, more empathetic web content
</p>

