"""
Site Empathy Analysis
=====================

Analyze website content for empathy using NLP.

This package combines Firecrawl web crawling with the Sharma et al. (2020) 
empathy analysis framework to measure emotional warmth, understanding, and 
engagement in web content.

Attribution
-----------
The empathy analysis framework is based on:

    Sharma, A., Miner, A.S., Atkins, D.C., & Althoff, T. (2020).
    A Computational Approach to Understanding Empathy Expressed in 
    Text-Based Mental Health Support. 
    Proceedings of EMNLP 2020.
    https://aclanthology.org/2020.emnlp-main.425/

Example
-------
>>> from site_empathy_analysis import SiteEmpathyAnalyzer
>>> analyzer = SiteEmpathyAnalyzer()
>>> results = analyzer.analyze_site("https://example.com")
>>> results.to_csv("empathy_report.csv")
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__citation__ = """
@inproceedings{sharma2020empathy,
    title={A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support},
    author={Sharma, Ashish and Miner, Adam S and Atkins, David C and Althoff, Tim},
    year={2020},
    booktitle={EMNLP}
}
"""

from site_empathy_analysis.analyzer import SiteEmpathyAnalyzer
from site_empathy_analysis.models.empathy_model import EmpathyScorer
from site_empathy_analysis.crawler import SiteCrawler

__all__ = [
    "SiteEmpathyAnalyzer",
    "EmpathyScorer", 
    "SiteCrawler",
    "__version__",
    "__citation__",
]

