"""
Site Empathy Analyzer
=====================

Main interface for analyzing website content for empathy.
Combines Firecrawl crawling with Sharma et al. (2020) empathy analysis.
"""

import os
import pandas as pd
from dataclasses import dataclass, field
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
        
        return site_analysis
    
    def _display_summary(self, analysis: SiteAnalysis):
        """Display analysis summary in terminal."""
        console.print()
        
        # Main stats table
        stats_table = Table(title=f"📊 Empathy Analysis: {analysis.domain}", show_header=False)
        stats_table.add_column("Metric", style="cyan", width=30)
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Pages analyzed", f"{analysis.page_count}")
        stats_table.add_row("", "")
        
        # Empathy score with color coding
        score = analysis.mean_empathy_score
        if score >= 0.3:
            score_style = "bold green"
        elif score >= 0.15:
            score_style = "yellow"
        else:
            score_style = "red"
        
        stats_table.add_row(
            "Mean Empathy Score", 
            f"[{score_style}]{score:.3f}[/] (0-1 scale)"
        )
        stats_table.add_row("Median Empathy Score", f"{analysis.median_empathy_score:.3f}")
        stats_table.add_row("Score Range", f"{analysis.min_empathy_score:.3f} - {analysis.max_empathy_score:.3f}")
        stats_table.add_row("", "")
        
        # Dimension breakdown
        stats_table.add_row("[bold]Dimension Breakdown[/]", "")
        stats_table.add_row(
            "  🔥 Emotional Warmth (ER)", 
            f"{analysis.pct_with_emotional_reaction:.1f}% of pages"
        )
        stats_table.add_row(
            "  🧠 Understanding (IP)", 
            f"{analysis.pct_with_interpretation:.1f}% of pages"
        )
        stats_table.add_row(
            "  💬 Engagement (EX)", 
            f"{analysis.pct_with_exploration:.1f}% of pages"
        )
        stats_table.add_row("", "")
        
        # Language analysis
        stats_table.add_row("[bold]Language Analysis[/]", "")
        stats_table.add_row(
            "  Empathic phrases found", 
            f"{analysis.total_empathic_phrases}"
        )
        stats_table.add_row(
            "  Non-empathic indicators", 
            f"{analysis.total_non_empathic_indicators}"
        )
        
        if analysis.most_common_empathic:
            stats_table.add_row(
                "  Top empathic phrases", 
                ", ".join(analysis.most_common_empathic[:3])
            )
        
        console.print(stats_table)
        
        # Key takeaways
        console.print()
        takeaways = self._generate_takeaways(analysis)
        
        takeaway_panel = Panel(
            "\n".join(f"• {t}" for t in takeaways),
            title="💡 Key Takeaways",
            border_style="yellow",
        )
        console.print(takeaway_panel)
    
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
        """Display summary across multiple sites."""
        console.print()
        
        summary_table = Table(title="📊 Batch Summary")
        summary_table.add_column("Domain", style="cyan")
        summary_table.add_column("Pages", justify="right")
        summary_table.add_column("Empathy", justify="right")
        summary_table.add_column("ER %", justify="right")
        summary_table.add_column("IP %", justify="right")
        summary_table.add_column("EX %", justify="right")
        
        for r in sorted(results, key=lambda x: x.mean_empathy_score, reverse=True):
            score = r.mean_empathy_score
            if score >= 0.3:
                score_str = f"[green]{score:.3f}[/]"
            elif score >= 0.15:
                score_str = f"[yellow]{score:.3f}[/]"
            else:
                score_str = f"[red]{score:.3f}[/]"
            
            summary_table.add_row(
                r.domain[:30],
                str(r.page_count),
                score_str,
                f"{r.pct_with_emotional_reaction:.0f}%",
                f"{r.pct_with_interpretation:.0f}%",
                f"{r.pct_with_exploration:.0f}%",
            )
        
        console.print(summary_table)


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

