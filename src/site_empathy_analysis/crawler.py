"""
Site Crawler Module
===================

Firecrawl-powered site crawling with progress tracking and resume capability.
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
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
)
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

console = Console()


@dataclass
class CrawledPage:
    """Represents a single crawled page."""
    
    url: str
    domain: str
    title: Optional[str] = None
    meta_description: Optional[str] = None
    markdown: Optional[str] = None
    html: Optional[str] = None
    word_count: int = 0
    crawled_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_analysis_text(self, max_length: int = 4000) -> str:
        """Get full text content suitable for empathy analysis."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.meta_description:
            parts.append(self.meta_description)
        if self.markdown:
            # Use the full markdown content (up to max_length)
            parts.append(self.markdown)
        return ' '.join(parts)[:max_length]
    
    def get_full_content(self) -> str:
        """Get the complete page content for analysis."""
        return self.markdown or ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            "url": self.url,
            "domain": self.domain,
            "title": self.title,
            "meta_description": self.meta_description,
            "word_count": self.word_count,
            "crawled_at": self.crawled_at,
            "text_preview": self.get_analysis_text()[:200],
        }


@dataclass 
class CrawlResult:
    """Results from crawling a site."""
    
    domain: str
    pages: List[CrawledPage]
    status: str  # success, empty, error, rate_limited
    error_message: Optional[str] = None
    crawl_time_seconds: float = 0.0
    
    @property
    def page_count(self) -> int:
        return len(self.pages)
    
    @property
    def total_words(self) -> int:
        return sum(p.word_count for p in self.pages)


class SiteCrawler:
    """
    Firecrawl-powered site crawler with progress tracking.
    
    Example
    -------
    >>> crawler = SiteCrawler(api_key="fc-...")
    >>> result = crawler.crawl("https://example.com", max_pages=50)
    >>> print(f"Crawled {result.page_count} pages")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_delay: float = 1.0,
        max_retries: int = 2,
    ):
        """
        Initialize the crawler.
        
        Args:
            api_key: Firecrawl API key. If None, looks for FIRECRAWL_API_KEY env var.
            rate_limit_delay: Seconds to wait between requests.
            max_retries: Number of retries on failure.
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._firecrawl = None
    
    @property
    def firecrawl(self):
        """Lazy-load Firecrawl client."""
        if self._firecrawl is None:
            if not self.api_key:
                raise ValueError(
                    "Firecrawl API key required. Set FIRECRAWL_API_KEY environment "
                    "variable or pass api_key to SiteCrawler."
                )
            try:
                from firecrawl import Firecrawl
                self._firecrawl = Firecrawl(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "firecrawl-py is required. Install with: pip install firecrawl-py"
                )
        return self._firecrawl
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    
    def _normalize_url(self, url: str) -> str:
        """Ensure URL has protocol."""
        if not url.startswith(("http://", "https://")):
            return f"https://{url}"
        return url
    
    def crawl(
        self,
        url: str,
        max_pages: int = 0,
        include_paths: Optional[str] = None,
        show_progress: bool = True,
        on_page: Optional[Callable[[CrawledPage], None]] = None,
    ) -> CrawlResult:
        """
        Crawl a single site.
        
        Args:
            url: Site URL or domain to crawl
            max_pages: Maximum pages to crawl (0 = unlimited)
            include_paths: Only crawl paths matching this pattern (e.g., "/blog/*")
            show_progress: Show progress in terminal
            on_page: Optional callback for each crawled page
            
        Returns:
            CrawlResult with all crawled pages
        """
        url = self._normalize_url(url)
        domain = self._extract_domain(url)
        
        start_time = time.time()
        pages = []
        error_message = None
        status = "success"
        
        # Build scope description
        scope_desc = "entire site" if max_pages == 0 else f"max {max_pages} pages"
        if include_paths:
            scope_desc = f"path: {include_paths}" + (f" ({max_pages} max)" if max_pages > 0 else "")
        
        if show_progress:
            console.print(f"\n[bold blue]🕷️  Crawling:[/] {domain}")
            console.print(f"   Scope: {scope_desc}")
        
        # Add delay to avoid rate limits
        time.sleep(self.rate_limit_delay)
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                disable=not show_progress,
            ) as progress:
                task = progress.add_task(f"Crawling {domain}...", total=None)
                
                # 0 = unlimited (use very high number)
                crawl_limit = max_pages if max_pages > 0 else 10000
                
                # Build crawl options
                crawl_kwargs = {
                    "url": url,
                    "limit": crawl_limit,
                    "scrape_options": {
                        "formats": ["markdown", "html"],
                        "onlyMainContent": False,
                    },
                    "poll_interval": 3,
                }
                
                # Add path filter if specified
                if include_paths:
                    # Ensure path ends with * for matching
                    path_pattern = include_paths if include_paths.endswith("*") else f"{include_paths.rstrip('/')}/*"
                    crawl_kwargs["includePaths"] = [path_pattern]
                
                result = self.firecrawl.crawl(**crawl_kwargs)
                
                progress.update(task, completed=True)
            
            if not hasattr(result, 'data') or not result.data:
                status = "empty"
                if show_progress:
                    console.print(f"   [yellow]⚠️  No pages found[/]")
            else:
                for doc in result.data:
                    # Extract URL
                    page_url = ""
                    if doc.metadata:
                        page_url = (
                            getattr(doc.metadata, 'source_url', '') or 
                            getattr(doc.metadata, 'url', '') or ''
                        )
                    
                    if not page_url:
                        continue
                    
                    markdown = getattr(doc, 'markdown', '') or ''
                    html = getattr(doc, 'html', '') or ''
                    
                    # Extract metadata
                    title = None
                    description = None
                    if doc.metadata:
                        title = getattr(doc.metadata, 'title', None)
                        description = getattr(doc.metadata, 'description', None)
                    
                    page = CrawledPage(
                        url=page_url,
                        domain=domain,
                        title=title,
                        meta_description=description,
                        markdown=markdown[:15000],  # Truncate for storage
                        html=html[:50000] if html else None,
                        word_count=len(markdown.split()) if markdown else 0,
                    )
                    
                    pages.append(page)
                    
                    if on_page:
                        on_page(page)
                
                if show_progress:
                    console.print(f"   [green]✓ Found {len(pages)} pages[/]")
                    
        except Exception as e:
            error_str = str(e).lower()
            
            if "rate" in error_str or "429" in error_str:
                status = "rate_limited"
                error_message = "Rate limited by Firecrawl"
            elif "credit" in error_str or "quota" in error_str:
                status = "credits_exhausted"
                error_message = "Firecrawl credits exhausted"
            else:
                status = "error"
                error_message = str(e)[:200]
            
            if show_progress:
                console.print(f"   [red]✗ Error: {error_message}[/]")
        
        crawl_time = time.time() - start_time
        
        return CrawlResult(
            domain=domain,
            pages=pages,
            status=status,
            error_message=error_message,
            crawl_time_seconds=crawl_time,
        )
    
    def crawl_batch(
        self,
        urls: List[str],
        max_pages_per_site: int = 500,
        show_progress: bool = True,
        on_site_complete: Optional[Callable[[CrawlResult], None]] = None,
    ) -> List[CrawlResult]:
        """
        Crawl multiple sites with progress tracking.
        
        Args:
            urls: List of URLs or domains to crawl
            max_pages_per_site: Max pages per site
            show_progress: Show progress display
            on_site_complete: Callback after each site completes
            
        Returns:
            List of CrawlResult objects
        """
        results = []
        total_sites = len(urls)
        
        if show_progress:
            console.print(Panel.fit(
                f"[bold]Crawling {total_sites} sites[/]\n"
                f"Max pages per site: {max_pages_per_site}",
                title="🕷️  Site Crawler",
                border_style="blue",
            ))
        
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=not show_progress,
        ) as progress:
            overall_task = progress.add_task(
                "Overall progress", 
                total=total_sites
            )
            
            for i, url in enumerate(urls):
                domain = self._extract_domain(self._normalize_url(url))
                progress.update(
                    overall_task, 
                    description=f"[{i+1}/{total_sites}] {domain}"
                )
                
                result = self.crawl(
                    url, 
                    max_pages=max_pages_per_site,
                    show_progress=False,
                )
                results.append(result)
                
                # Status indicator
                if result.status == "success":
                    status_icon = "[green]✓[/]"
                elif result.status == "empty":
                    status_icon = "[yellow]○[/]"
                else:
                    status_icon = "[red]✗[/]"
                
                if show_progress:
                    console.print(
                        f"  {status_icon} {domain[:40]:<40} │ "
                        f"{result.page_count:>3} pages │ "
                        f"{result.crawl_time_seconds:.1f}s"
                    )
                
                if on_site_complete:
                    on_site_complete(result)
                
                progress.update(overall_task, advance=1)
        
        # Summary
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.status == "success")
        total_pages = sum(r.page_count for r in results)
        
        if show_progress:
            console.print()
            summary_table = Table(title="Crawl Summary", show_header=False)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")
            
            summary_table.add_row("Sites crawled", f"{successful}/{total_sites}")
            summary_table.add_row("Total pages", f"{total_pages:,}")
            summary_table.add_row("Time elapsed", f"{timedelta(seconds=int(total_time))}")
            summary_table.add_row("Avg pages/site", f"{total_pages/max(1, successful):.1f}")
            
            console.print(summary_table)
        
        return results

