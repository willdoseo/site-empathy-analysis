"""
Site Empathy Analyzer
=====================

Main interface for analyzing website content for empathy.
Combines Firecrawl crawling with Sharma et al. (2020) empathy analysis.
"""

import os
import json
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from site_empathy_analysis.crawler import SiteCrawler, CrawlResult, CrawledPage
from site_empathy_analysis.models.empathy_model import (
    EmpathyScorer, 
    EmpathyResult,
    EMPATHY_DIMENSIONS,
)

console = Console()


@dataclass
class PageAnalysis:
    """Combined crawl and empathy analysis for a single page."""
    
    url: str
    domain: str
    title: Optional[str]
    meta_description: Optional[str]
    word_count: int
    
    # Empathy scores
    empathy_score: float
    emotional_reaction_score: float
    interpretation_score: float
    exploration_score: float
    
    # Empathy levels (0, 1, 2)
    er_level: int
    ip_level: int
    ex_level: int
    
    # Language analysis
    empathic_phrases: List[str]
    non_empathic_indicators: List[str]
    empathic_phrase_count: int
    non_empathic_count: int
    
    # Optional fields with defaults (must come last)
    content_preview: str = ""  # First ~500 chars of actual page content
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            "url": self.url,
            "domain": self.domain,
            "title": self.title or "",
            "meta_description": self.meta_description or "",
            "content_preview": self.content_preview,
            "word_count": self.word_count,
            "empathy_score": round(self.empathy_score, 4),
            "emotional_reaction_score": round(self.emotional_reaction_score, 4),
            "interpretation_score": round(self.interpretation_score, 4),
            "exploration_score": round(self.exploration_score, 4),
            "er_level": self.er_level,
            "ip_level": self.ip_level,
            "ex_level": self.ex_level,
            "empathic_phrases": "|".join(self.empathic_phrases),
            "non_empathic_indicators": "|".join(self.non_empathic_indicators),
            "empathic_phrase_count": self.empathic_phrase_count,
            "non_empathic_count": self.non_empathic_count,
            "analyzed_at": self.analyzed_at,
        }
    
    def get_grade(self) -> str:
        """Get letter grade for empathy score."""
        score = self.empathy_score
        if score >= 0.4:
            return "A"
        elif score >= 0.3:
            return "B"
        elif score >= 0.2:
            return "C"
        elif score >= 0.1:
            return "D"
        else:
            return "F"
    
    def get_empathy_profile(self) -> str:
        """Get a human-readable empathy profile."""
        has_er = self.er_level >= 1
        has_ip = self.ip_level >= 1
        has_ex = self.ex_level >= 1
        
        if has_er and has_ip and has_ex:
            return "🌟 Full Empathy (Warm + Understanding + Engaging)"
        elif has_er and has_ip:
            return "💚 Empathic (Warm + Understanding)"
        elif has_er and has_ex:
            return "💛 Warm & Engaging"
        elif has_ip and has_ex:
            return "💙 Understanding & Engaging"
        elif has_er:
            return "🟠 Emotionally Warm Only"
        elif has_ip:
            return "🔵 Understanding Only (Clinical)"
        elif has_ex:
            return "🟣 Engaging Only"
        else:
            return "⚪ Minimal Empathy"


@dataclass
class SiteAnalysis:
    """Complete analysis results for a site."""
    
    domain: str
    url: str
    pages: List[PageAnalysis]
    
    # Aggregate scores
    mean_empathy_score: float
    median_empathy_score: float
    max_empathy_score: float
    min_empathy_score: float
    
    # Dimension breakdowns
    pct_with_emotional_reaction: float
    pct_with_interpretation: float
    pct_with_exploration: float
    
    # Language stats
    total_empathic_phrases: int
    total_non_empathic_indicators: int
    most_common_empathic: List[str]
    most_common_non_empathic: List[str]
    
    # Metadata
    crawl_status: str
    crawl_time_seconds: float
    analysis_time_seconds: float
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def page_count(self) -> int:
        return len(self.pages)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert page analyses to DataFrame."""
        return pd.DataFrame([p.to_dict() for p in self.pages])
    
    def to_csv(self, path: Union[str, Path]) -> Path:
        """Export page-level analysis to CSV."""
        path = Path(path)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path
    
    def to_json(self, path: Union[str, Path], indent: int = 2) -> Path:
        """Export complete analysis to JSON."""
        path = Path(path)
        data = {
            "summary": self.summary_dict(),
            "pages": [p.to_dict() for p in self.pages],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent)
        return path
    
    def to_html(self, path: Union[str, Path]) -> Path:
        """Generate comprehensive HTML report with charts."""
        path = Path(path)
        html = self._generate_html_report()
        with open(path, 'w') as f:
            f.write(html)
        return path
    
    def summary_dict(self) -> dict:
        """Get summary statistics as dictionary."""
        return {
            "domain": self.domain,
            "url": self.url,
            "page_count": self.page_count,
            "mean_empathy_score": round(self.mean_empathy_score, 4),
            "median_empathy_score": round(self.median_empathy_score, 4),
            "max_empathy_score": round(self.max_empathy_score, 4),
            "min_empathy_score": round(self.min_empathy_score, 4),
            "pct_with_emotional_reaction": round(self.pct_with_emotional_reaction, 2),
            "pct_with_interpretation": round(self.pct_with_interpretation, 2),
            "pct_with_exploration": round(self.pct_with_exploration, 2),
            "total_empathic_phrases": self.total_empathic_phrases,
            "total_non_empathic_indicators": self.total_non_empathic_indicators,
            "most_common_empathic": "|".join(self.most_common_empathic[:5]),
            "most_common_non_empathic": "|".join(self.most_common_non_empathic[:5]),
            "crawl_status": self.crawl_status,
            "crawl_time_seconds": round(self.crawl_time_seconds, 2),
            "analysis_time_seconds": round(self.analysis_time_seconds, 2),
            "analyzed_at": self.analyzed_at,
        }
    
    def get_score_distribution(self) -> Dict[str, int]:
        """Get distribution of pages by score range."""
        dist = {"F (0-0.1)": 0, "D (0.1-0.2)": 0, "C (0.2-0.3)": 0, "B (0.3-0.4)": 0, "A (0.4+)": 0}
        for page in self.pages:
            grade = page.get_grade()
            if grade == "F":
                dist["F (0-0.1)"] += 1
            elif grade == "D":
                dist["D (0.1-0.2)"] += 1
            elif grade == "C":
                dist["C (0.2-0.3)"] += 1
            elif grade == "B":
                dist["B (0.3-0.4)"] += 1
            else:
                dist["A (0.4+)"] += 1
        return dist
    
    def get_top_pages(self, n: int = 5) -> List[PageAnalysis]:
        """Get top N pages by empathy score."""
        return sorted(self.pages, key=lambda p: p.empathy_score, reverse=True)[:n]
    
    def get_bottom_pages(self, n: int = 5) -> List[PageAnalysis]:
        """Get bottom N pages by empathy score."""
        return sorted(self.pages, key=lambda p: p.empathy_score)[:n]
    
    def _generate_html_report(self) -> str:
        """Generate comprehensive HTML report with charts."""
        # Get data for charts
        score_dist = self.get_score_distribution()
        top_pages = self.get_top_pages(10)
        bottom_pages = self.get_bottom_pages(10)
        
        # Prepare chart data
        score_labels = list(score_dist.keys())
        score_values = list(score_dist.values())
        
        # Dimension data
        dim_labels = ["Emotional Warmth", "Understanding", "Engagement"]
        dim_values = [
            self.pct_with_emotional_reaction,
            self.pct_with_interpretation,
            self.pct_with_exploration
        ]
        
        # Page scores for histogram
        page_scores = [p.empathy_score for p in self.pages]
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Empathy Analysis: {self.domain}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        .header p {{
            font-size: 1.2em;
            opacity: 0.95;
        }}
        .content {{
            padding: 40px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .grade {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.2em;
        }}
        .grade-A {{ background: #10b981; color: white; }}
        .grade-B {{ background: #3b82f6; color: white; }}
        .grade-C {{ background: #f59e0b; color: white; }}
        .grade-D {{ background: #ef4444; color: white; }}
        .grade-F {{ background: #991b1b; color: white; }}
        .section {{
            margin-bottom: 50px;
        }}
        .section-title {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 20px;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .chart-container {{
            background: #f9fafb;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .chart-wrapper {{
            position: relative;
            height: 400px;
        }}
        .page-list {{
            display: grid;
            gap: 15px;
        }}
        .page-card {{
            background: #f9fafb;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            transition: all 0.2s;
        }}
        .page-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateX(5px);
        }}
        .page-card.low-score {{
            border-left-color: #ef4444;
        }}
        .page-title {{
            font-weight: 600;
            font-size: 1.1em;
            margin-bottom: 8px;
            color: #111;
        }}
        .page-url {{
            font-size: 0.85em;
            color: #667eea;
            margin-bottom: 12px;
            word-break: break-all;
        }}
        .page-score {{
            display: inline-block;
            padding: 4px 12px;
            background: #667eea;
            color: white;
            border-radius: 12px;
            font-size: 0.9em;
            font-weight: 600;
            margin-right: 10px;
        }}
        .page-profile {{
            font-size: 0.9em;
            color: #666;
        }}
        .insights {{
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            padding: 25px;
            border-radius: 12px;
            border-left: 5px solid #f59e0b;
        }}
        .insights h3 {{
            color: #92400e;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .insights ul {{
            list-style: none;
            padding-left: 0;
        }}
        .insights li {{
            padding: 8px 0;
            color: #78350f;
            font-size: 1.05em;
        }}
        .insights li:before {{
            content: "💡 ";
            margin-right: 8px;
        }}
        .phrase-badge {{
            display: inline-block;
            background: #dbeafe;
            color: #1e40af;
            padding: 6px 12px;
            border-radius: 16px;
            margin: 4px;
            font-size: 0.9em;
        }}
        .footer {{
            background: #f9fafb;
            padding: 30px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #e5e7eb;
        }}
        .citation {{
            font-style: italic;
            margin-top: 10px;
            color: #999;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Empathy Analysis Report</h1>
            <p>{self.domain}</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Analyzed: {self.analyzed_at[:10]} • {self.page_count} pages</p>
        </div>
        
        <div class="content">
            <!-- Key Metrics -->
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Overall Grade</div>
                    <div class="metric-value">
                        <span class="grade grade-{self._get_overall_grade()}">{self._get_overall_grade()}</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Empathy</div>
                    <div class="metric-value">{self.mean_empathy_score:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Median Empathy</div>
                    <div class="metric-value">{self.median_empathy_score:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Pages Analyzed</div>
                    <div class="metric-value">{self.page_count}</div>
                </div>
            </div>
            
            <!-- Key Insights -->
            <div class="section">
                <div class="insights">
                    <h3>🎯 Key Insights & Recommendations</h3>
                    <ul>
                        {self._generate_html_insights()}
                    </ul>
                </div>
            </div>
            
            <!-- Charts -->
            <div class="section">
                <h2 class="section-title">📊 Score Distribution</h2>
                <div class="chart-container">
                    <div class="chart-wrapper">
                        <canvas id="gradeChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">🎭 Empathy Dimensions</h2>
                <div class="chart-container">
                    <div class="chart-wrapper">
                        <canvas id="dimensionChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">📈 Score Distribution Histogram</h2>
                <div class="chart-container">
                    <div class="chart-wrapper">
                        <canvas id="histogramChart"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Top Pages -->
            <div class="section">
                <h2 class="section-title">🌟 Top Performing Pages</h2>
                <div class="page-list">
                    {self._generate_page_cards(top_pages, is_top=True)}
                </div>
            </div>
            
            <!-- Bottom Pages -->
            <div class="section">
                <h2 class="section-title">⚠️ Pages Needing Attention</h2>
                <div class="page-list">
                    {self._generate_page_cards(bottom_pages, is_top=False)}
                </div>
            </div>
            
            <!-- Language Analysis -->
            <div class="section">
                <h2 class="section-title">💬 Language Analysis</h2>
                <div style="background: #f9fafb; padding: 25px; border-radius: 12px;">
                    <h3 style="color: #10b981; margin-bottom: 15px;">✅ Most Common Empathic Phrases</h3>
                    <div>
                        {self._generate_phrase_badges(self.most_common_empathic[:15])}
                    </div>
                    <h3 style="color: #ef4444; margin-top: 30px; margin-bottom: 15px;">❌ Non-Empathic Indicators</h3>
                    <div>
                        {self._generate_phrase_badges(self.most_common_non_empathic[:15])}
                    </div>
                    <div style="margin-top: 25px; padding: 15px; background: white; border-radius: 8px;">
                        <strong>Summary:</strong> Found {self.total_empathic_phrases} empathic phrases 
                        and {self.total_non_empathic_indicators} non-empathic indicators across all pages.
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Analysis Method:</strong> Sharma et al. (2020) Empathy Framework</p>
            <p class="citation">
                Sharma, A., Miner, A.S., Atkins, D.C., & Althoff, T. (2020).<br>
                A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support.<br>
                EMNLP 2020.
            </p>
            <p style="margin-top: 15px;">Generated by Site Empathy Analysis • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
    
    <script>
        // Grade Distribution Chart
        new Chart(document.getElementById('gradeChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(score_labels)},
                datasets: [{{
                    label: 'Number of Pages',
                    data: {json.dumps(score_values)},
                    backgroundColor: [
                        'rgba(153, 27, 27, 0.8)',
                        'rgba(239, 68, 68, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(59, 130, 246, 0.8)',
                        'rgba(16, 185, 129, 0.8)'
                    ],
                    borderColor: [
                        'rgb(153, 27, 27)',
                        'rgb(239, 68, 68)',
                        'rgb(245, 158, 11)',
                        'rgb(59, 130, 246)',
                        'rgb(16, 185, 129)'
                    ],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{
                        display: true,
                        text: 'Page Distribution by Empathy Grade',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{ stepSize: 1 }}
                    }}
                }}
            }}
        }});
        
        // Dimension Chart
        new Chart(document.getElementById('dimensionChart'), {{
            type: 'radar',
            data: {{
                labels: {json.dumps(dim_labels)},
                datasets: [{{
                    label: '% of Pages',
                    data: {json.dumps(dim_values)},
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgb(102, 126, 234)',
                    borderWidth: 3,
                    pointBackgroundColor: 'rgb(102, 126, 234)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgb(102, 126, 234)',
                    pointRadius: 5
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Empathy Dimension Coverage',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            stepSize: 20,
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});
        
        // Histogram
        new Chart(document.getElementById('histogramChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps([f"{i/10:.1f}" for i in range(11)])},
                datasets: [{{
                    label: 'Number of Pages',
                    data: {json.dumps(self._get_histogram_data(page_scores))},
                    backgroundColor: 'rgba(102, 126, 234, 0.7)',
                    borderColor: 'rgb(102, 126, 234)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{
                        display: true,
                        text: 'Empathy Score Distribution (0.0 - 1.0)',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{ stepSize: 1 }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Empathy Score'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        return html
    
    def _get_overall_grade(self) -> str:
        """Get overall grade for the site."""
        score = self.mean_empathy_score
        if score >= 0.4:
            return "A"
        elif score >= 0.3:
            return "B"
        elif score >= 0.2:
            return "C"
        elif score >= 0.1:
            return "D"
        else:
            return "F"
    
    def _get_histogram_data(self, scores: List[float]) -> List[int]:
        """Get histogram bins for score distribution."""
        bins = [0] * 11  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0, 1.0+
        for score in scores:
            idx = min(int(score * 10), 10)
            bins[idx] += 1
        return bins
    
    def _generate_html_insights(self) -> str:
        """Generate HTML list items for insights."""
        insights = []
        
        er_pct = self.pct_with_emotional_reaction
        ip_pct = self.pct_with_interpretation
        ex_pct = self.pct_with_exploration
        
        has_warmth = er_pct >= 20
        has_understanding = ip_pct >= 20
        
        if has_warmth and has_understanding:
            insights.append("<li><strong>Excellent foundation:</strong> Content demonstrates both emotional warmth and understanding.</li>")
        elif has_warmth:
            insights.append("<li><strong>Add depth:</strong> Good emotional warmth, but consider adding more explanatory content to show deeper understanding.</li>")
        elif has_understanding:
            insights.append("<li><strong>Warm it up:</strong> Content is informative but could use more emotional connection. Add phrases like 'we understand', 'you're not alone'.</li>")
        else:
            insights.append("<li><strong>Major opportunity:</strong> Content lacks both warmth and understanding. This is the primary area for improvement.</li>")
        
        if er_pct < 10:
            insights.append("<li><strong>Missing emotional warmth:</strong> Consider adding supportive phrases that acknowledge feelings and show care.</li>")
        
        if ex_pct < 10:
            insights.append("<li><strong>Low engagement:</strong> Add more questions and dialogue-inducing language to encourage interaction.</li>")
        
        if self.total_empathic_phrases > 0:
            insights.append(f"<li><strong>What's working:</strong> Found {self.total_empathic_phrases} empathic phrases. Top ones: {', '.join(self.most_common_empathic[:3])}</li>")
        
        grade = self._get_overall_grade()
        if grade in ['A', 'B']:
            insights.append("<li><strong>Strong performance:</strong> Your content demonstrates good empathy. Focus on consistency across all pages.</li>")
        elif grade == 'C':
            insights.append("<li><strong>Room for growth:</strong> Average empathy detected. Focus on the specific recommendations above.</li>")
        else:
            insights.append("<li><strong>Significant improvement needed:</strong> Low empathy scores indicate transactional or clinical tone. Humanize your content.</li>")
        
        return "\n".join(insights)
    
    def _generate_page_cards(self, pages: List[PageAnalysis], is_top: bool) -> str:
        """Generate HTML for page cards."""
        cards = []
        for i, page in enumerate(pages, 1):
            low_class = "" if is_top else "low-score"
            title = page.title or "Untitled Page"
            preview = page.content_preview[:150] + "..." if len(page.content_preview) > 150 else page.content_preview
            
            cards.append(f"""
                <div class="page-card {low_class}">
                    <div class="page-title">{i}. {title}</div>
                    <div class="page-url">{page.url}</div>
                    <div>
                        <span class="page-score">Score: {page.empathy_score:.3f}</span>
                        <span class="page-profile">{page.get_empathy_profile()}</span>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                        ER: {page.er_level} | IP: {page.ip_level} | EX: {page.ex_level} | 
                        {page.empathic_phrase_count} empathic phrases
                    </div>
                    {f'<div style="margin-top: 12px; padding: 10px; background: white; border-radius: 4px; font-size: 0.85em; color: #555;"><strong>Content preview:</strong> {preview}</div>' if preview else ''}
                </div>
            """)
        
        return "\n".join(cards)
    
    def _generate_phrase_badges(self, phrases: List[str]) -> str:
        """Generate HTML badges for phrases."""
        if not phrases:
            return '<span style="color: #999;">None detected</span>'
        return "\n".join(f'<span class="phrase-badge">{phrase}</span>' for phrase in phrases)


class SiteEmpathyAnalyzer:
    """
    Main interface for site empathy analysis.
    
    Combines Firecrawl site crawling with the Sharma et al. (2020) 
    empathy analysis framework.
    
    Example
    -------
    >>> analyzer = SiteEmpathyAnalyzer(firecrawl_key="fc-...")
    >>> result = analyzer.analyze_site("https://example.com")
    >>> result.to_csv("empathy_report.csv")
    >>> print(f"Mean empathy: {result.mean_empathy_score:.2f}")
    
    Attribution
    -----------
    The empathy analysis is based on:
    
        Sharma, A., Miner, A.S., Atkins, D.C., & Althoff, T. (2020).
        A Computational Approach to Understanding Empathy Expressed in 
        Text-Based Mental Health Support. EMNLP 2020.
    """
    
    def __init__(
        self,
        firecrawl_key: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the analyzer.
        
        Args:
            firecrawl_key: Firecrawl API key. Uses FIRECRAWL_API_KEY env var if None.
            model_path: Path to trained empathy model weights.
            device: Device for inference ('cuda', 'mps', 'cpu'). Auto-detected if None.
        """
        self.firecrawl_key = firecrawl_key or os.getenv("FIRECRAWL_API_KEY")
        self.model_path = model_path
        self.device = device
        
        self._crawler = None
        self._scorer = None
    
    @property
    def crawler(self) -> SiteCrawler:
        """Lazy-load crawler."""
        if self._crawler is None:
            self._crawler = SiteCrawler(api_key=self.firecrawl_key)
        return self._crawler
    
    @property
    def scorer(self) -> EmpathyScorer:
        """Lazy-load empathy scorer."""
        if self._scorer is None:
            self._scorer = EmpathyScorer(
                model_path=self.model_path,
                device=self.device,
            )
        return self._scorer
    
    def set_api_key(self, key: str):
        """Set Firecrawl API key programmatically."""
        self.firecrawl_key = key
        self._crawler = None  # Reset to use new key
    
    def _analyze_page(self, page: CrawledPage) -> PageAnalysis:
        """Analyze a single crawled page for empathy using full content."""
        # Get the full page content
        full_text = page.get_analysis_text(max_length=10000)
        
        if len(full_text.strip()) < 20:
            # Not enough text to analyze
            return PageAnalysis(
                url=page.url,
                domain=page.domain,
                title=page.title,
                meta_description=page.meta_description,
                word_count=page.word_count,
                content_preview=full_text[:500] if full_text else "",
                empathy_score=0.0,
                emotional_reaction_score=0.0,
                interpretation_score=0.0,
                exploration_score=0.0,
                er_level=0,
                ip_level=0,
                ex_level=0,
                empathic_phrases=[],
                non_empathic_indicators=["insufficient_text"],
                empathic_phrase_count=0,
                non_empathic_count=1,
            )
        
        # Analyze content in chunks and aggregate scores
        # This ensures we analyze the FULL page, not just the first 64 tokens
        chunk_size = 500  # characters per chunk
        chunks = []
        
        # Split into overlapping chunks
        for i in range(0, len(full_text), chunk_size - 50):
            chunk = full_text[i:i + chunk_size]
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        
        if not chunks:
            chunks = [full_text[:500]]
        
        # Analyze each chunk
        chunk_results = []
        for chunk in chunks[:20]:  # Limit to 20 chunks max
            try:
                result = self.scorer.score(chunk)
                chunk_results.append(result)
            except:
                pass
        
        if not chunk_results:
            # Fallback to single analysis
            result = self.scorer.score(full_text[:500])
            chunk_results = [result]
        
        # Aggregate scores across chunks (use max to capture peak empathy)
        avg_empathy = sum(r.empathy_score for r in chunk_results) / len(chunk_results)
        max_empathy = max(r.empathy_score for r in chunk_results)
        avg_er = sum(r.emotional_reaction for r in chunk_results) / len(chunk_results)
        avg_ip = sum(r.interpretation for r in chunk_results) / len(chunk_results)
        avg_ex = sum(r.exploration for r in chunk_results) / len(chunk_results)
        
        # Use weighted average favoring max (captures best empathic content)
        final_empathy = (avg_empathy * 0.6) + (max_empathy * 0.4)
        
        # Get max levels
        max_er_level = max(r.er_level for r in chunk_results)
        max_ip_level = max(r.ip_level for r in chunk_results)
        max_ex_level = max(r.ex_level for r in chunk_results)
        
        # Collect all empathic phrases from full text
        all_empathic = []
        all_non_empathic = []
        for r in chunk_results:
            all_empathic.extend(r.empathic_phrases)
            all_non_empathic.extend(r.non_empathic_indicators)
        
        # Deduplicate
        all_empathic = list(set(all_empathic))
        all_non_empathic = list(set(all_non_empathic))
        
        # Get full content that was analyzed (cleaned up for CSV)
        content_preview = ""
        if page.markdown:
            # Include full analyzed content (up to 10k chars to match what was analyzed)
            content = page.markdown[:10000].replace('\n', ' ').replace('\r', ' ')
            content = ' '.join(content.split())  # Normalize whitespace
            content_preview = content
        
        return PageAnalysis(
            url=page.url,
            domain=page.domain,
            title=page.title,
            meta_description=page.meta_description,
            word_count=page.word_count,
            content_preview=content_preview,
            empathy_score=final_empathy,
            emotional_reaction_score=avg_er,
            interpretation_score=avg_ip,
            exploration_score=avg_ex,
            er_level=max_er_level,
            ip_level=max_ip_level,
            ex_level=max_ex_level,
            empathic_phrases=all_empathic,
            non_empathic_indicators=all_non_empathic,
            empathic_phrase_count=len(all_empathic),
            non_empathic_count=len(all_non_empathic),
        )
    
    def _calculate_site_stats(
        self, 
        pages: List[PageAnalysis],
        domain: str,
        url: str,
        crawl_result: CrawlResult,
        analysis_time: float,
    ) -> SiteAnalysis:
        """Calculate aggregate statistics for a site."""
        import statistics
        from collections import Counter
        
        if not pages:
            return SiteAnalysis(
                domain=domain,
                url=url,
                pages=[],
                mean_empathy_score=0.0,
                median_empathy_score=0.0,
                max_empathy_score=0.0,
                min_empathy_score=0.0,
                pct_with_emotional_reaction=0.0,
                pct_with_interpretation=0.0,
                pct_with_exploration=0.0,
                total_empathic_phrases=0,
                total_non_empathic_indicators=0,
                most_common_empathic=[],
                most_common_non_empathic=[],
                crawl_status=crawl_result.status,
                crawl_time_seconds=crawl_result.crawl_time_seconds,
                analysis_time_seconds=analysis_time,
            )
        
        scores = [p.empathy_score for p in pages]
        
        # Count phrases
        all_empathic = []
        all_non_empathic = []
        for p in pages:
            all_empathic.extend(p.empathic_phrases)
            all_non_empathic.extend(p.non_empathic_indicators)
        
        empathic_counter = Counter(all_empathic)
        non_empathic_counter = Counter(all_non_empathic)
        
        return SiteAnalysis(
            domain=domain,
            url=url,
            pages=pages,
            mean_empathy_score=statistics.mean(scores),
            median_empathy_score=statistics.median(scores),
            max_empathy_score=max(scores),
            min_empathy_score=min(scores),
            pct_with_emotional_reaction=sum(1 for p in pages if p.er_level >= 1) / len(pages) * 100,
            pct_with_interpretation=sum(1 for p in pages if p.ip_level >= 1) / len(pages) * 100,
            pct_with_exploration=sum(1 for p in pages if p.ex_level >= 1) / len(pages) * 100,
            total_empathic_phrases=len(all_empathic),
            total_non_empathic_indicators=len(all_non_empathic),
            most_common_empathic=[phrase for phrase, _ in empathic_counter.most_common(10)],
            most_common_non_empathic=[phrase for phrase, _ in non_empathic_counter.most_common(10)],
            crawl_status=crawl_result.status,
            crawl_time_seconds=crawl_result.crawl_time_seconds,
            analysis_time_seconds=analysis_time,
        )
    
    def analyze_site(
        self,
        url: str,
        max_pages: int = 0,
        include_paths: Optional[str] = None,
        show_progress: bool = True,
    ) -> SiteAnalysis:
        """
        Analyze a website for empathy.
        
        Args:
            url: Website URL or domain
            max_pages: Maximum pages to crawl (0 = unlimited)
            include_paths: Only crawl paths matching this pattern (e.g., "/blog/")
            show_progress: Show progress in terminal
            
        Returns:
            SiteAnalysis with page-level and aggregate results
        """
        import time
        
        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        domain = urlparse(url).netloc.replace("www.", "")
        
        if show_progress:
            console.print()
            console.print(Panel.fit(
                f"[bold]Analyzing:[/] {domain}\n"
                f"[dim]Max pages: {max_pages}[/]",
                title="🔍 Site Empathy Analysis",
                border_style="cyan",
            ))
        
        # Step 1: Crawl the site
        if show_progress:
            console.print("\n[bold cyan]Step 1/2:[/] Crawling site...")
        
        crawl_result = self.crawler.crawl(
            url, 
            max_pages=max_pages,
            include_paths=include_paths,
            show_progress=show_progress
        )
        
        if crawl_result.status != "success" or not crawl_result.pages:
            if show_progress:
                console.print(f"[yellow]⚠️  Crawl failed or returned no pages: {crawl_result.status}[/]")
            
            return self._calculate_site_stats(
                [], domain, url, crawl_result, 0.0
            )
        
        # Step 2: Analyze pages for empathy
        if show_progress:
            console.print(f"\n[bold cyan]Step 2/2:[/] Analyzing {len(crawl_result.pages)} pages for empathy...")
        
        analysis_start = time.time()
        page_analyses = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=not show_progress,
        ) as progress:
            task = progress.add_task(
                "Analyzing empathy...", 
                total=len(crawl_result.pages)
            )
            
            for page in crawl_result.pages:
                try:
                    analysis = self._analyze_page(page)
                    page_analyses.append(analysis)
                except Exception as e:
                    # Create empty analysis on error
                    page_analyses.append(PageAnalysis(
                        url=page.url,
                        domain=page.domain,
                        title=page.title,
                        meta_description=page.meta_description,
                        word_count=page.word_count,
                        empathy_score=0.0,
                        emotional_reaction_score=0.0,
                        interpretation_score=0.0,
                        exploration_score=0.0,
                        er_level=0,
                        ip_level=0,
                        ex_level=0,
                        empathic_phrases=[],
                        non_empathic_indicators=[f"error:{str(e)[:50]}"],
                        empathic_phrase_count=0,
                        non_empathic_count=1,
                    ))
                
                progress.update(task, advance=1)
        
        analysis_time = time.time() - analysis_start
        
        # Calculate site-level statistics
        site_analysis = self._calculate_site_stats(
            page_analyses, domain, url, crawl_result, analysis_time
        )
        
        # Display summary
        if show_progress:
            self._display_summary(site_analysis)
            console.print()
            console.print(f"[bold green]✅ Analysis complete![/] Total time: {crawl_result.crawl_time_seconds + analysis_time:.1f}s")
        
        return site_analysis
    
    def _display_summary(self, analysis: SiteAnalysis):
        """Display enhanced analysis summary in terminal."""
        console.print()
        
        # Header with overall grade
        grade = analysis._get_overall_grade()
        grade_colors = {"A": "green", "B": "blue", "C": "yellow", "D": "red", "F": "red"}
        grade_color = grade_colors.get(grade, "white")
        
        header_text = Text()
        header_text.append(f"📊 Empathy Analysis: {analysis.domain}\n", style="bold cyan")
        header_text.append(f"Overall Grade: ", style="white")
        header_text.append(f"{grade}", style=f"bold {grade_color}")
        header_text.append(f" | Pages: {analysis.page_count} | Mean Score: {analysis.mean_empathy_score:.3f}", style="dim")
        
        console.print(Panel(header_text, border_style="cyan"))
        console.print()
        
        # Main stats table
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column("Metric", style="cyan", width=32)
        stats_table.add_column("Value", style="white")
        
        # Empathy score with color coding and bar
        score = analysis.mean_empathy_score
        if score >= 0.3:
            score_style = "bold green"
        elif score >= 0.15:
            score_style = "yellow"
        else:
            score_style = "red"
        
        score_bar = self._create_bar(score, max_value=1.0, width=20, filled_char="█", empty_char="░")
        
        stats_table.add_row(
            "Mean Empathy Score", 
            f"[{score_style}]{score:.3f}[/] {score_bar}"
        )
        stats_table.add_row("Median Score", f"{analysis.median_empathy_score:.3f}")
        stats_table.add_row("Range", f"{analysis.min_empathy_score:.3f} → {analysis.max_empathy_score:.3f}")
        stats_table.add_row("", "")
        
        # Dimension breakdown with bars
        stats_table.add_row("[bold underline]Empathy Dimensions[/]", "")
        
        er_bar = self._create_bar(analysis.pct_with_emotional_reaction, max_value=100, width=20)
        stats_table.add_row(
            "  🔥 Emotional Warmth (ER)", 
            f"{analysis.pct_with_emotional_reaction:5.1f}% {er_bar}"
        )
        
        ip_bar = self._create_bar(analysis.pct_with_interpretation, max_value=100, width=20)
        stats_table.add_row(
            "  🧠 Understanding (IP)", 
            f"{analysis.pct_with_interpretation:5.1f}% {ip_bar}"
        )
        
        ex_bar = self._create_bar(analysis.pct_with_exploration, max_value=100, width=20)
        stats_table.add_row(
            "  💬 Engagement (EX)", 
            f"{analysis.pct_with_exploration:5.1f}% {ex_bar}"
        )
        stats_table.add_row("", "")
        
        # Language analysis
        stats_table.add_row("[bold underline]Language Analysis[/]", "")
        stats_table.add_row(
            "  ✅ Empathic phrases", 
            f"[green]{analysis.total_empathic_phrases}[/]"
        )
        stats_table.add_row(
            "  ❌ Non-empathic indicators", 
            f"[red]{analysis.total_non_empathic_indicators}[/]"
        )
        
        if analysis.most_common_empathic:
            stats_table.add_row(
                "  🌟 Top phrases", 
                f"[dim]{', '.join(analysis.most_common_empathic[:3])}[/]"
            )
        
        console.print(stats_table)
        
        # Score distribution histogram
        console.print()
        console.print("[bold cyan]Score Distribution:[/]")
        self._display_histogram(analysis)
        
        # Top and bottom pages
        console.print()
        self._display_top_bottom_pages(analysis)
        
        # Key takeaways
        console.print()
        takeaways = self._generate_takeaways(analysis)
        
        takeaway_panel = Panel(
            "\n".join(f"• {t}" for t in takeaways),
            title="💡 Key Insights & Recommendations",
            border_style="yellow",
        )
        console.print(takeaway_panel)
        
        # Export suggestions
        console.print()
        export_text = Text()
        export_text.append("💾 Export Options:\n", style="bold cyan")
        export_text.append("  • ", style="dim")
        export_text.append("result.to_csv('report.csv')", style="green")
        export_text.append("  - Page-level data\n", style="dim")
        export_text.append("  • ", style="dim")
        export_text.append("result.to_html('report.html')", style="green")
        export_text.append("  - Interactive dashboard\n", style="dim")
        export_text.append("  • ", style="dim")
        export_text.append("result.to_json('report.json')", style="green")
        export_text.append("  - Complete analysis data", style="dim")
        console.print(Panel(export_text, border_style="dim"))
    
    def _create_bar(self, value: float, max_value: float = 100, width: int = 20, 
                    filled_char: str = "█", empty_char: str = "░") -> str:
        """Create a text-based progress bar."""
        filled = int((value / max_value) * width)
        empty = width - filled
        
        # Color based on value percentage
        pct = (value / max_value) * 100
        if pct >= 70:
            color = "green"
        elif pct >= 40:
            color = "yellow"
        else:
            color = "red"
        
        bar = f"[{color}]{filled_char * filled}[/]{empty_char * empty}"
        return bar
    
    def _display_histogram(self, analysis: SiteAnalysis):
        """Display ASCII histogram of score distribution."""
        dist = analysis.get_score_distribution()
        
        if not analysis.pages:
            console.print("[dim]No pages to display[/]")
            return
        
        max_count = max(dist.values()) if dist.values() else 1
        
        hist_table = Table(show_header=False, box=None, padding=(0, 1))
        hist_table.add_column("Grade", style="cyan", width=15)
        hist_table.add_column("Bar", style="white")
        hist_table.add_column("Count", justify="right", style="white", width=6)
        
        grade_colors = {
            "F (0-0.1)": "red",
            "D (0.1-0.2)": "red",
            "C (0.2-0.3)": "yellow",
            "B (0.3-0.4)": "blue",
            "A (0.4+)": "green"
        }
        
        for grade_range, count in dist.items():
            if count > 0:
                bar_length = int((count / max_count) * 30)
                color = grade_colors.get(grade_range, "white")
                bar = f"[{color}]{'█' * bar_length}[/]"
                hist_table.add_row(grade_range, bar, str(count))
        
        console.print(hist_table)
    
    def _display_top_bottom_pages(self, analysis: SiteAnalysis):
        """Display top and bottom performing pages."""
        top_pages = analysis.get_top_pages(5)
        bottom_pages = analysis.get_bottom_pages(5)
        
        # Top pages
        console.print("[bold green]🌟 Top 5 Pages:[/]")
        top_table = Table(show_header=True, box=None, padding=(0, 1))
        top_table.add_column("Score", justify="right", style="green", width=8)
        top_table.add_column("Grade", style="green", width=7)
        top_table.add_column("Profile", style="white", width=25)
        top_table.add_column("Page", style="cyan")
        
        for page in top_pages:
            title = page.title or "Untitled"
            if len(title) > 50:
                title = title[:47] + "..."
            top_table.add_row(
                f"{page.empathy_score:.3f}",
                page.get_grade(),
                page.get_empathy_profile()[:25],
                title
            )
        
        console.print(top_table)
        console.print()
        
        # Bottom pages
        console.print("[bold red]⚠️  Bottom 5 Pages:[/]")
        bottom_table = Table(show_header=True, box=None, padding=(0, 1))
        bottom_table.add_column("Score", justify="right", style="red", width=8)
        bottom_table.add_column("Grade", style="red", width=7)
        bottom_table.add_column("Profile", style="white", width=25)
        bottom_table.add_column("Page", style="cyan")
        
        for page in bottom_pages:
            title = page.title or "Untitled"
            if len(title) > 50:
                title = title[:47] + "..."
            bottom_table.add_row(
                f"{page.empathy_score:.3f}",
                page.get_grade(),
                page.get_empathy_profile()[:25],
                title
            )
        
        console.print(bottom_table)
    
    def _generate_takeaways(self, analysis: SiteAnalysis) -> List[str]:
        """Generate key takeaways from the analysis."""
        takeaways = []
        
        score = analysis.mean_empathy_score
        er_pct = analysis.pct_with_emotional_reaction
        ip_pct = analysis.pct_with_interpretation
        ex_pct = analysis.pct_with_exploration
        
        # Overall empathy assessment based on ACTUAL dimensions present
        has_warmth = er_pct >= 20
        has_understanding = ip_pct >= 20
        has_engagement = ex_pct >= 10
        
        # Build accurate description
        if has_warmth and has_understanding:
            takeaways.append("✅ HIGH empathy - content shows both emotional warmth AND understanding")
        elif has_warmth and not has_understanding:
            takeaways.append("🟡 PARTIAL empathy - content has warmth but lacks depth of understanding")
        elif has_understanding and not has_warmth:
            takeaways.append("🟠 CLINICAL tone - content shows understanding but LACKS emotional warmth")
        else:
            takeaways.append("🔴 LOW empathy - content lacks both warmth and understanding")
        
        # Dimension-specific insights
        if er_pct < 10:
            takeaways.append("🔥 Missing emotional warmth (ER: {:.0f}%) - add phrases like 'we care', 'you're not alone', 'we understand how hard this is'".format(er_pct))
        
        if ip_pct > 30 and er_pct < 15:
            takeaways.append("📊 'Understanding without warmth' pattern - pages explain problems but don't express care or compassion")
        
        if ex_pct < 5:
            takeaways.append("💬 Low engagement (EX: {:.0f}%) - consider adding questions like 'How can we help?' or 'What are you experiencing?'".format(ex_pct))
        
        # Language suggestions
        if analysis.total_non_empathic_indicators > analysis.total_empathic_phrases:
            takeaways.append("📝 More clinical/transactional language than empathic - consider humanizing copy")
        
        if analysis.most_common_empathic:
            takeaways.append(f"🌟 Empathic phrases found: {', '.join(analysis.most_common_empathic[:5])}")
        elif analysis.total_empathic_phrases == 0:
            takeaways.append("⚠️ No empathic phrases detected - consider adding: 'your journey', 'we understand', 'you deserve', 'here for you'")
        
        return takeaways
    
    def analyze_batch(
        self,
        urls: List[str],
        max_pages_per_site: int = 500,
        show_progress: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[SiteAnalysis]:
        """
        Analyze multiple websites.
        
        Args:
            urls: List of URLs or domains
            max_pages_per_site: Max pages per site
            show_progress: Show progress display
            output_dir: Directory to save individual CSVs (optional)
            
        Returns:
            List of SiteAnalysis objects
        """
        results = []
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        if show_progress:
            console.print(Panel.fit(
                f"[bold]Analyzing {len(urls)} sites[/]\n"
                f"Max pages per site: {max_pages_per_site}",
                title="🔍 Batch Empathy Analysis",
                border_style="cyan",
            ))
        
        for i, url in enumerate(urls):
            if show_progress:
                console.print(f"\n[bold]Site {i+1}/{len(urls)}[/]")
            
            analysis = self.analyze_site(
                url, 
                max_pages=max_pages_per_site,
                show_progress=show_progress,
            )
            results.append(analysis)
            
            # Save individual CSV if output_dir provided
            if output_dir:
                csv_path = output_dir / f"{analysis.domain.replace('.', '_')}_empathy.csv"
                analysis.to_csv(csv_path)
                if show_progress:
                    console.print(f"   💾 Saved: {csv_path}")
        
        # Summary across all sites
        if show_progress and len(results) > 1:
            self._display_batch_summary(results)
        
        return results
    
    def _display_batch_summary(self, results: List[SiteAnalysis]):
        """Display enhanced summary across multiple sites."""
        console.print()
        console.print()
        
        # Overall batch stats
        total_pages = sum(r.page_count for r in results)
        avg_score = sum(r.mean_empathy_score for r in results) / len(results) if results else 0
        
        header = Panel(
            f"[bold]Batch Analysis Complete[/]\n"
            f"Sites: {len(results)} | Total Pages: {total_pages} | Average Score: {avg_score:.3f}",
            title="📊 Batch Summary",
            border_style="cyan"
        )
        console.print(header)
        console.print()
        
        # Detailed comparison table
        summary_table = Table(title="Site-by-Site Comparison (Ranked by Empathy Score)")
        summary_table.add_column("Rank", justify="center", style="dim", width=6)
        summary_table.add_column("Domain", style="cyan", width=30)
        summary_table.add_column("Grade", justify="center", width=7)
        summary_table.add_column("Score", justify="right", width=8)
        summary_table.add_column("Pages", justify="right", width=7)
        summary_table.add_column("ER", justify="right", width=6)
        summary_table.add_column("IP", justify="right", width=6)
        summary_table.add_column("EX", justify="right", width=6)
        summary_table.add_column("Profile", style="white", width=25)
        
        for i, r in enumerate(sorted(results, key=lambda x: x.mean_empathy_score, reverse=True), 1):
            score = r.mean_empathy_score
            if score >= 0.3:
                score_str = f"[green]{score:.3f}[/]"
            elif score >= 0.15:
                score_str = f"[yellow]{score:.3f}[/]"
            else:
                score_str = f"[red]{score:.3f}[/]"
            
            grade = r._get_overall_grade()
            grade_colors = {"A": "green", "B": "blue", "C": "yellow", "D": "red", "F": "red"}
            grade_str = f"[{grade_colors.get(grade, 'white')}]{grade}[/]"
            
            # Get profile from top page
            profile = "Mixed"
            if r.pages:
                top_page = max(r.pages, key=lambda p: p.empathy_score)
                profile = top_page.get_empathy_profile()[:25]
            
            summary_table.add_row(
                f"#{i}",
                r.domain[:28],
                grade_str,
                score_str,
                str(r.page_count),
                f"{r.pct_with_emotional_reaction:.0f}%",
                f"{r.pct_with_interpretation:.0f}%",
                f"{r.pct_with_exploration:.0f}%",
                profile,
            )
        
        console.print(summary_table)
        
        # Key insights
        console.print()
        best_site = max(results, key=lambda x: x.mean_empathy_score)
        worst_site = min(results, key=lambda x: x.mean_empathy_score)
        
        insights = []
        insights.append(f"🏆 Highest empathy: [bold]{best_site.domain}[/] ({best_site.mean_empathy_score:.3f})")
        insights.append(f"📉 Needs improvement: [bold]{worst_site.domain}[/] ({worst_site.mean_empathy_score:.3f})")
        
        # Count by grade
        grade_counts = {}
        for r in results:
            grade = r._get_overall_grade()
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        grade_summary = ", ".join([f"{count}×{grade}" for grade, count in sorted(grade_counts.items())])
        insights.append(f"📊 Grade distribution: {grade_summary}")
        
        insights_panel = Panel(
            "\n".join(f"• {i}" for i in insights),
            title="🎯 Batch Insights",
            border_style="yellow"
        )
        console.print(insights_panel)


def export_batch_to_csv(
    results: List[SiteAnalysis],
    output_path: Union[str, Path],
    include_pages: bool = True,
) -> Path:
    """
    Export batch analysis results to CSV.
    
    Args:
        results: List of SiteAnalysis objects
        output_path: Output CSV path
        include_pages: If True, exports page-level data. If False, site summaries only.
        
    Returns:
        Path to created CSV file
    """
    output_path = Path(output_path)
    
    if include_pages:
        # Page-level export
        all_pages = []
        for r in results:
            for p in r.pages:
                page_dict = p.to_dict()
                all_pages.append(page_dict)
        
        df = pd.DataFrame(all_pages)
    else:
        # Site summary export
        summaries = [r.summary_dict() for r in results]
        df = pd.DataFrame(summaries)
    
    df.to_csv(output_path, index=False)
    return output_path

