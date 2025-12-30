"""Tests for the SiteEmpathyAnalyzer."""

import pytest
from unittest.mock import Mock, patch


class TestSiteEmpathyAnalyzer:
    """Tests for the main analyzer class."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        from site_empathy_analysis.analyzer import SiteEmpathyAnalyzer
        
        analyzer = SiteEmpathyAnalyzer(firecrawl_key="test-key")
        
        assert analyzer.firecrawl_key == "test-key"
    
    def test_set_api_key(self):
        """Test setting API key programmatically."""
        from site_empathy_analysis.analyzer import SiteEmpathyAnalyzer
        
        analyzer = SiteEmpathyAnalyzer()
        analyzer.set_api_key("new-key")
        
        assert analyzer.firecrawl_key == "new-key"


class TestPageAnalysis:
    """Tests for PageAnalysis dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from site_empathy_analysis.analyzer import PageAnalysis
        
        page = PageAnalysis(
            url="https://example.com",
            domain="example.com",
            title="Test Page",
            meta_description="A test page",
            word_count=100,
            empathy_score=0.5,
            emotional_reaction_score=0.4,
            interpretation_score=0.3,
            exploration_score=0.2,
            er_level=1,
            ip_level=1,
            ex_level=0,
            empathic_phrases=["we understand"],
            non_empathic_indicators=["click here"],
            empathic_phrase_count=1,
            non_empathic_count=1,
        )
        
        d = page.to_dict()
        
        assert d["url"] == "https://example.com"
        assert d["empathy_score"] == 0.5
        assert d["empathic_phrase_count"] == 1


class TestSiteAnalysis:
    """Tests for SiteAnalysis dataclass."""
    
    def test_summary_dict(self):
        """Test summary dictionary generation."""
        from site_empathy_analysis.analyzer import SiteAnalysis, PageAnalysis
        
        pages = [
            PageAnalysis(
                url="https://example.com/1",
                domain="example.com",
                title="Page 1",
                meta_description="",
                word_count=100,
                empathy_score=0.3,
                emotional_reaction_score=0.2,
                interpretation_score=0.3,
                exploration_score=0.1,
                er_level=1,
                ip_level=1,
                ex_level=0,
                empathic_phrases=[],
                non_empathic_indicators=[],
                empathic_phrase_count=0,
                non_empathic_count=0,
            ),
            PageAnalysis(
                url="https://example.com/2",
                domain="example.com",
                title="Page 2",
                meta_description="",
                word_count=200,
                empathy_score=0.5,
                emotional_reaction_score=0.4,
                interpretation_score=0.4,
                exploration_score=0.2,
                er_level=2,
                ip_level=1,
                ex_level=1,
                empathic_phrases=["we understand"],
                non_empathic_indicators=[],
                empathic_phrase_count=1,
                non_empathic_count=0,
            ),
        ]
        
        site = SiteAnalysis(
            domain="example.com",
            url="https://example.com",
            pages=pages,
            mean_empathy_score=0.4,
            median_empathy_score=0.4,
            max_empathy_score=0.5,
            min_empathy_score=0.3,
            pct_with_emotional_reaction=100.0,
            pct_with_interpretation=100.0,
            pct_with_exploration=50.0,
            total_empathic_phrases=1,
            total_non_empathic_indicators=0,
            most_common_empathic=["we understand"],
            most_common_non_empathic=[],
            crawl_status="success",
            crawl_time_seconds=5.0,
            analysis_time_seconds=2.0,
        )
        
        summary = site.summary_dict()
        
        assert summary["domain"] == "example.com"
        assert summary["page_count"] == 2
        assert summary["mean_empathy_score"] == 0.4

