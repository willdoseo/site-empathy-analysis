"""
Progress Display Utilities
==========================

Pretty progress tracking for terminal output using Rich.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

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
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

console = Console()


@dataclass
class ProgressStats:
    """Track progress statistics."""
    
    total: int = 0
    completed: int = 0
    successful: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    @property
    def rate(self) -> float:
        if self.elapsed == 0:
            return 0
        return self.completed / self.elapsed
    
    @property
    def eta(self) -> float:
        if self.rate == 0:
            return 0
        remaining = self.total - self.completed
        return remaining / self.rate
    
    @property
    def success_rate(self) -> float:
        if self.completed == 0:
            return 0
        return self.successful / self.completed * 100
    
    def format_elapsed(self) -> str:
        return str(timedelta(seconds=int(self.elapsed)))
    
    def format_eta(self) -> str:
        if self.eta == 0:
            return "calculating..."
        return str(timedelta(seconds=int(self.eta)))


class ProgressTracker:
    """
    Track and display progress for multi-step operations.
    
    Example
    -------
    >>> tracker = ProgressTracker(total=100, description="Analyzing pages")
    >>> with tracker:
    ...     for page in pages:
    ...         process(page)
    ...         tracker.advance()
    """
    
    def __init__(
        self,
        total: int,
        description: str = "Processing",
        show_rate: bool = True,
        show_eta: bool = True,
        console: Optional[Console] = None,
    ):
        self.stats = ProgressStats(total=total)
        self.description = description
        self.show_rate = show_rate
        self.show_eta = show_eta
        self.console = console or Console()
        self._progress = None
        self._task = None
    
    def __enter__(self):
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._progress.__enter__()
        self._task = self._progress.add_task(self.description, total=self.stats.total)
        return self
    
    def __exit__(self, *args):
        if self._progress:
            self._progress.__exit__(*args)
    
    def advance(self, success: bool = True):
        """Advance progress by one."""
        self.stats.completed += 1
        if success:
            self.stats.successful += 1
        else:
            self.stats.failed += 1
        
        if self._progress and self._task is not None:
            self._progress.update(self._task, advance=1)
    
    def update_description(self, description: str):
        """Update the progress description."""
        if self._progress and self._task is not None:
            self._progress.update(self._task, description=description)


def create_progress_display(
    total: int,
    description: str = "Processing",
) -> Progress:
    """
    Create a rich progress display.
    
    Args:
        total: Total items to process
        description: Description text
        
    Returns:
        Progress context manager
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def display_summary_table(
    title: str,
    data: Dict[str, Any],
    style: str = "cyan",
) -> None:
    """
    Display a summary table.
    
    Args:
        title: Table title
        data: Dictionary of metric -> value pairs
        style: Color style for metrics
    """
    table = Table(title=title, show_header=False)
    table.add_column("Metric", style=style)
    table.add_column("Value", style="green")
    
    for key, value in data.items():
        if isinstance(value, float):
            value = f"{value:.3f}"
        table.add_row(str(key), str(value))
    
    console.print(table)


def display_comparison_table(
    title: str,
    items: List[Dict[str, Any]],
    columns: List[str],
    sort_by: Optional[str] = None,
    reverse: bool = True,
) -> None:
    """
    Display a comparison table.
    
    Args:
        title: Table title
        items: List of dictionaries with data
        columns: Column names to display
        sort_by: Column to sort by
        reverse: Sort descending if True
    """
    if sort_by and sort_by in columns:
        items = sorted(items, key=lambda x: x.get(sort_by, 0), reverse=reverse)
    
    table = Table(title=title)
    
    for col in columns:
        table.add_column(col.replace("_", " ").title())
    
    for item in items:
        row = []
        for col in columns:
            value = item.get(col, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            row.append(str(value))
        table.add_row(*row)
    
    console.print(table)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_banner(title: str, subtitle: Optional[str] = None, style: str = "cyan"):
    """Print a styled banner."""
    content = f"[bold]{title}[/]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/]"
    
    console.print(Panel.fit(content, border_style=style))


def print_success(message: str):
    """Print a success message."""
    console.print(f"[green]✓[/] {message}")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[yellow]⚠️[/] {message}")


def print_error(message: str):
    """Print an error message."""
    console.print(f"[red]✗[/] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"[blue]ℹ[/] {message}")

