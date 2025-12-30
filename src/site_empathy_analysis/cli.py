"""
Site Empathy Analysis - Voight-Kampff Edition
==============================================

"The tortoise lays on its back, its belly baking in the hot sun..."
A tool to measure the empathy of digital entities.
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.style import Style
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Brand colors - Basic ANSI colors for Mac Terminal compatibility
CORAL = "yellow"        # Primary accent (closest to orange in basic ANSI)
TEAL = "cyan"           # Secondary
CREAM = "white"         # Background tone
SLATE = "white"         # Text

# Blade Runner / Electric Sheep quotes for flavor
QUOTES = [
    # Classic Blade Runner / Electric Sheep references
    '"The tortoise lays on its back, its belly baking in the hot sun..."',
    '"More human than human" is our motto.',
    "Do androids dream of electric sheep?",
    "I've seen things you people wouldn't believe...",
    "All those moments will be lost in time, like tears in rain.",
    "The light that burns twice as bright burns half as long.",
    # Empathy-themed originals
    "Empathy is the algorithm that makes us human.",
    "Behind every click is a person seeking connection.",
    "The best content speaks to the heart, not just the mind.",
    "Words have the power to heal or to alienate.",
    "Measuring empathy: where data science meets emotional intelligence.",
    "Your website's empathy score: more human than human?",
]

# ASCII art banner - Clean and simple
BANNER = f"""
[bold {CORAL}]
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃                                                                 ┃
  ┃   ███████╗██╗████████╗███████╗                                  ┃
  ┃   ██╔════╝██║╚══██╔══╝██╔════╝                                  ┃
  ┃   ███████╗██║   ██║   █████╗                                    ┃
  ┃   ╚════██║██║   ██║   ██╔══╝                                    ┃
  ┃   ███████║██║   ██║   ███████╗                                  ┃
  ┃   ╚══════╝╚═╝   ╚═╝   ╚══════╝                                  ┃
  ┃                                                                 ┃
  ┃   ███████╗███╗   ███╗██████╗  █████╗ ████████╗██╗  ██╗██╗   ██╗ ┃
  ┃   ██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝██║  ██║╚██╗ ██╔╝ ┃
  ┃   █████╗  ██╔████╔██║██████╔╝███████║   ██║   ███████║ ╚████╔╝  ┃
  ┃   ██╔══╝  ██║╚██╔╝██║██╔═══╝ ██╔══██║   ██║   ██╔══██║  ╚██╔╝   ┃
  ┃   ███████╗██║ ╚═╝ ██║██║     ██║  ██║   ██║   ██║  ██║   ██║    ┃
  ┃   ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝    ┃
  ┃                                                                 ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
[/]
[{TEAL}]          ◆ Do Androids Dream of Empathic Content? ◆[/]
[dim]               Measuring the humanity of your website[/]
"""

MINI_BANNER = f"""
[bold {CORAL}]┌────────────────────────────────────┐
│  ⚡🐑  SITE EMPATHY ANALYSIS       │
└────────────────────────────────────┘[/]
"""


def show_quote():
    """Display a random Blade Runner quote."""
    quote = random.choice(QUOTES)
    console.print(f"\n[dim italic]{quote}[/]\n")


def electric_sheep_loading(message: str = "Analyzing"):
    """Show a themed loading animation."""
    frames = ["◐", "◓", "◑", "◒"]
    sheep = ["🐑", "⚡🐑", "🐑⚡", "⚡🐑⚡"]
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn(f"[{TEAL}]{message}...[/]"),
        console=console,
    )


def get_saved_api_key() -> Optional[str]:
    """Get API key from config file."""
    config_path = Path.home() / ".config" / "site-empathy" / "config"
    if config_path.exists():
        with open(config_path) as f:
            for line in f:
                if line.startswith("FIRECRAWL_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return None


def ensure_models_downloaded():
    """Ensure empathy models are downloaded before analysis."""
    from site_empathy_analysis.models.empathy_model import MODEL_CACHE_DIR, MODEL_BASE_URL
    import requests
    from tqdm import tqdm
    
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    models = ["modern_ER.pth", "modern_IP.pth", "modern_EX.pth"]
    missing = [m for m in models if not (MODEL_CACHE_DIR / m).exists()]
    
    if not missing:
        return True  # All models present
    
    console.print()
    console.print(Panel(
        f"[bold {CORAL}]📥 NEURAL CORE INITIALIZATION[/]\n\n"
        f"[dim]First-run setup: Downloading empathy analysis models (~3GB total)\n"
        f"This only happens once - models will be cached for future use.[/]\n\n"
        f"[dim italic]\"I've seen things you people wouldn't believe...\"[/]",
        border_style=CORAL
    ))
    console.print()
    
    for filename in missing:
        dim = filename.replace("modern_", "").replace(".pth", "")
        url = f"{MODEL_BASE_URL}/{filename}"
        cache_path = MODEL_CACHE_DIR / filename
        
        console.print(f"[{TEAL}]Downloading {dim} model...[/]")
        console.print(f"[dim]   {url}[/]")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(cache_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"   {dim}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            console.print(f"[{TEAL}]   ✓ {dim} model ready[/]")
            console.print()
            
        except requests.exceptions.RequestException as e:
            console.print(f"[{CORAL}]   ✗ Download failed: {e}[/]")
            if cache_path.exists():
                cache_path.unlink()
            console.print(f"[{CORAL}]Cannot proceed without models. Check your internet connection.[/]")
            sys.exit(1)
    
    console.print(f"[bold {TEAL}]✓ All neural cores initialized[/]")
    console.print()
    return True


def save_api_key(key: str):
    """Save API key to config file."""
    config_dir = Path.home() / ".config" / "site-empathy"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / "config"
    with open(config_path, "w") as f:
        f.write(f"FIRECRAWL_API_KEY={key}\n")


def get_api_key_interactive() -> str:
    """Get API key interactively."""
    # Check environment variable first
    env_key = os.getenv("FIRECRAWL_API_KEY")
    if env_key:
        console.print(f"[{TEAL}]✓[/] Using API key from environment variable")
        return env_key
    
    # Check saved config
    saved_key = get_saved_api_key()
    if saved_key:
        use_saved = Confirm.ask(
            f"[{CORAL}]Found saved API key:[/] {saved_key[:8]}...{saved_key[-4:]}. Use it?",
            default=True
        )
        if use_saved:
            return saved_key
    
    # Prompt for key
    console.print()
    console.print(Panel(
        f"[bold {CORAL}]🔑 Firecrawl API Key Required[/]\n\n"
        f"[dim]The Voight-Kampff machine requires authentication.\n"
        f"Get your free API key at: [link]https://firecrawl.dev[/link][/]",
        border_style=CORAL
    ))
    console.print()
    
    key = Prompt.ask(f"[{TEAL}]Enter your Firecrawl API key[/]")
    
    if not key:
        console.print(f"[{CORAL}]API key is required. The test cannot proceed.[/]")
        sys.exit(1)
    
    # Offer to save
    save = Confirm.ask("[dim]Save this key for future tests?[/]", default=True)
    if save:
        save_api_key(key)
        console.print(f"[{TEAL}]✓[/] API key saved to ~/.config/site-empathy/config")
    
    return key


def display_result_summary(result):
    """Display empathy results with Blade Runner theming."""
    score = result.site_stats.get("avg_empathy_score", 0)
    
    # Determine empathy verdict
    if score >= 0.35:
        verdict = "HUMAN"
        verdict_color = "green"
        verdict_msg = "High empathy detected. Subject appears human."
        icon = "🟢"
    elif score >= 0.20:
        verdict = "INCONCLUSIVE"
        verdict_color = "yellow"
        verdict_msg = "Moderate empathy. Further testing recommended."
        icon = "🟡"
    else:
        verdict = "REPLICANT"
        verdict_color = CORAL
        verdict_msg = "Low empathy indicators. Possible replicant."
        icon = "🔴"
    
    # Results panel
    console.print()
    console.print(Panel(
        f"[bold]VOIGHT-KAMPFF TEST RESULTS[/]\n"
        f"[dim]Subject: {result.domain}[/]\n\n"
        f"  {icon} Verdict: [bold {verdict_color}]{verdict}[/]\n"
        f"  [dim]{verdict_msg}[/]\n\n"
        f"  [bold]Empathy Score:[/] [{TEAL}]{score:.3f}[/] / 1.000\n\n"
        f"  [bold]Dimension Breakdown:[/]\n"
        f"  ├─ 🔥 Emotional Reaction: {result.site_stats.get('avg_er_score', 0):.3f}\n"
        f"  ├─ 🧠 Interpretation:     {result.site_stats.get('avg_ip_score', 0):.3f}\n"
        f"  └─ 💬 Exploration:        {result.site_stats.get('avg_ex_score', 0):.3f}\n\n"
        f"  [dim]Pages analyzed: {result.site_stats.get('total_pages', 0)}[/]",
        title=f"[{CORAL}]◆ TEST COMPLETE ◆[/]",
        border_style=TEAL,
    ))


def run_interactive():
    """Run the interactive Voight-Kampff wizard."""
    console.clear()
    console.print(BANNER)
    show_quote()
    
    # Step 1: API Key
    console.print(Panel(
        f"[bold]PHASE 1:[/] Authentication\n"
        f"[dim]Initializing Voight-Kampff apparatus...[/]",
        border_style=CORAL
    ))
    api_key = get_api_key_interactive()
    console.print()
    
    # Step 2: Target
    console.print(Panel(
        f"[bold]PHASE 2:[/] Subject Selection\n"
        f"[dim]Identify the digital entity to be tested...[/]",
        border_style=CORAL
    ))
    console.print()
    domain = Prompt.ask(
        f"[{TEAL}]Enter website URL or domain[/]",
        default="example.com"
    )
    console.print()
    
    # Step 3: Parameters
    console.print(Panel(
        f"[bold]PHASE 3:[/] Test Parameters\n"
        f"[dim]Configure the empathy response measurement...[/]",
        border_style=CORAL
    ))
    console.print()
    
    # Crawl scope selection
    console.print(f"[{TEAL}]Crawl scope:[/]")
    console.print(f"  [dim]1.[/] Entire site (all pages)")
    console.print(f"  [dim]2.[/] Specific folder (e.g., /blog/)")
    console.print(f"  [dim]3.[/] Limited pages")
    console.print()
    
    scope_choice = Prompt.ask(
        f"[{TEAL}]Select crawl scope[/]",
        choices=["1", "2", "3"],
        default="1"
    )
    
    max_pages = 0  # Default unlimited
    include_paths = None
    
    if scope_choice == "2":
        # Specific folder
        include_paths = Prompt.ask(
            f"[{TEAL}]Enter path to crawl[/] [dim](e.g., /blog/ or /services/)[/]",
            default="/blog/"
        )
        max_pages = IntPrompt.ask(
            f"[{TEAL}]Max pages in this folder[/] [dim](0 = all)[/]",
            default=0
        )
    elif scope_choice == "3":
        # Limited pages
        max_pages = IntPrompt.ask(
            f"[{TEAL}]Maximum pages to analyze[/]",
            default=100
        )
    
    console.print()
    
    # Default output filename
    clean_domain = domain.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
    default_output = f"{clean_domain.replace('.', '_')}_empathy_report.csv"
    
    output_path = Prompt.ask(
        f"[{TEAL}]Output report filename[/]",
        default=default_output
    )
    
    console.print()
    
    # Build scope description
    if scope_choice == "1":
        scope_desc = "Entire site"
    elif scope_choice == "2":
        scope_desc = f"Folder: {include_paths}" + (f" (max {max_pages})" if max_pages > 0 else "")
    else:
        scope_desc = f"Limited to {max_pages} pages"
    
    # Confirm
    console.print(Panel(
        f"[bold]COMMENCING EMPATHY ANALYSIS[/]\n\n"
        f"  [bold]Subject:[/]     [{TEAL}]{domain}[/]\n"
        f"  [bold]Scope:[/]       [{TEAL}]{scope_desc}[/]\n"
        f"  [bold]Report:[/]      [{TEAL}]{output_path}[/]\n\n"
        f"[dim italic]\"Describe in single words only the good things\n"
        f" that come into your mind about your mother...\"[/]",
        title=f"[{CORAL}]◆ READY ◆[/]",
        border_style=TEAL
    ))
    console.print()
    
    if not Confirm.ask(f"[bold {TEAL}]Begin empathy test?[/]", default=True):
        console.print(f"[{CORAL}]Test aborted. Subject released.[/]")
        return
    
    console.print()
    console.print(f"[{TEAL}]Initializing pupillary response tracking...[/]")
    console.print()
    
    # Ensure models are downloaded before analysis
    ensure_models_downloaded()
    
    # Run the analysis
    try:
        from site_empathy_analysis.analyzer import SiteEmpathyAnalyzer
        
        analyzer = SiteEmpathyAnalyzer(firecrawl_key=api_key)
        result = analyzer.analyze_site(
            domain, 
            max_pages=max_pages, 
            include_paths=include_paths,
            show_progress=True
        )
        
        # Save output
        output_file = Path(output_path)
        result.to_csv(output_file)
        
        # Display results
        display_result_summary(result)
        
        console.print()
        console.print(f"[{TEAL}]✓[/] Full report saved to: [bold]{output_file.absolute()}[/]")
        
        # Random closing quote
        show_quote()
        
        # Ask if they want to test another
        console.print()
        if Confirm.ask("[dim]Test another subject?[/]", default=False):
            run_interactive()
            
    except Exception as e:
        console.print(f"\n[{CORAL}]❌ Test malfunction: {e}[/]")
        console.print(f"[dim]The Voight-Kampff machine has encountered an error.[/]")
        sys.exit(1)


@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version="0.1.0", prog_name="site-empathy")
def main(ctx):
    """
    🔬 Voight-Kampff Empathy Analysis System
    
    Analyze websites for empathy using the framework from Sharma et al. (2020).
    
    Run without arguments for interactive mode, or use subcommands.
    
    "More human than human" is our motto.
    """
    if ctx.invoked_subcommand is None:
        run_interactive()


@main.command()
@click.argument("url")
@click.option("-o", "--output", type=click.Path(), help="Output CSV file path")
@click.option("--max-pages", default=1000, help="Maximum pages to analyze (default: 1000, 0=unlimited)")
@click.option("--quiet", is_flag=True, help="Minimal output")
@click.option("--key", envvar="FIRECRAWL_API_KEY", help="Firecrawl API key")
def analyze(url: str, output: Optional[str], max_pages: int, quiet: bool, key: Optional[str]):
    """
    Run empathy test on a single website (non-interactive).
    
    URL can be a full URL or just a domain name.
    """
    if not quiet:
        console.print(MINI_BANNER)
    
    # Get API key
    api_key = key or os.getenv("FIRECRAWL_API_KEY") or get_saved_api_key()
    
    if not api_key:
        api_key = get_api_key_interactive()
    
    # Ensure models are downloaded before analysis
    ensure_models_downloaded()
    
    try:
        from site_empathy_analysis.analyzer import SiteEmpathyAnalyzer
        
        analyzer = SiteEmpathyAnalyzer(firecrawl_key=api_key)
        result = analyzer.analyze_site(url, max_pages=max_pages, show_progress=not quiet)
        
        # Save output
        if output:
            output_path = Path(output)
            result.to_csv(output_path)
            console.print(f"\n[{TEAL}]✓ Report saved to:[/] {output_path}")
        else:
            default_output = f"{result.domain.replace('.', '_')}_voightkampff.csv"
            result.to_csv(default_output)
            console.print(f"\n[{TEAL}]✓ Report saved to:[/] {default_output}")
        
        if not quiet:
            display_result_summary(result)
            
    except Exception as e:
        console.print(f"[{CORAL}]Test malfunction: {e}[/]")
        sys.exit(1)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), required=True, help="Output directory for CSVs")
@click.option("--max-pages", default=500, help="Maximum pages per site (default: 500, 0=unlimited)")
@click.option("--key", envvar="FIRECRAWL_API_KEY", help="Firecrawl API key")
def batch(input_file: str, output: str, max_pages: int, key: Optional[str]):
    """Run empathy tests on multiple websites from a file."""
    console.print(MINI_BANNER)
    console.print(f"[{TEAL}]Batch Voight-Kampff testing initiated...[/]")
    console.print()
    
    # Read URLs from file
    with open(input_file) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    if not urls:
        console.print(f"[{CORAL}]No URLs found in input file[/]")
        sys.exit(1)
    
    console.print(f"[{TEAL}]Found {len(urls)} subjects to test[/]")
    
    # Get API key
    api_key = key or os.getenv("FIRECRAWL_API_KEY") or get_saved_api_key()
    if not api_key:
        api_key = get_api_key_interactive()
    
    # Ensure models are downloaded before analysis
    ensure_models_downloaded()
    
    try:
        from site_empathy_analysis.analyzer import SiteEmpathyAnalyzer, export_batch_to_csv
        
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analyzer = SiteEmpathyAnalyzer(firecrawl_key=api_key)
        results = analyzer.analyze_batch(
            urls, 
            max_pages_per_site=max_pages,
            show_progress=True,
            output_dir=output_dir,
        )
        
        # Save combined CSV
        combined_path = output_dir / "all_subjects_voightkampff.csv"
        export_batch_to_csv(results, combined_path, include_pages=True)
        console.print(f"[{TEAL}]✓ Combined report saved to:[/] {combined_path}")
        
    except Exception as e:
        console.print(f"[{CORAL}]Error: {e}[/]")
        sys.exit(1)


@main.command()
@click.option("--set-key", help="Set Firecrawl API key")
@click.option("--show-key", is_flag=True, help="Show saved API key")
@click.option("--clear-key", is_flag=True, help="Remove saved API key")
def config(set_key: Optional[str], show_key: bool, clear_key: bool):
    """Configure the Voight-Kampff apparatus."""
    config_dir = Path.home() / ".config" / "site-empathy"
    config_path = config_dir / "config"
    
    if set_key:
        save_api_key(set_key)
        console.print(f"[{TEAL}]✓ API key saved[/]")
        return
    
    if show_key:
        key = get_saved_api_key()
        if key:
            masked = key[:8] + "*" * (len(key) - 12) + key[-4:] if len(key) > 15 else "***"
            console.print(f"API key: {masked}")
        else:
            console.print(f"[{CORAL}]No API key configured[/]")
        return
    
    if clear_key:
        if config_path.exists():
            config_path.unlink()
            console.print(f"[{TEAL}]✓ API key removed[/]")
        else:
            console.print(f"[{CORAL}]No saved key to remove[/]")
        return
    
    # Show current config
    console.print(Panel(
        f"[bold]Configuration Directory:[/] {config_dir}\n"
        f"[bold]API Key Status:[/] {'Configured' if get_saved_api_key() else 'Not configured'}",
        title=f"[{CORAL}]◆ APPARATUS SETTINGS ◆[/]",
        border_style=TEAL
    ))


@main.command()
@click.option("--download", is_flag=True, help="Download models if not present")
@click.option("--status", is_flag=True, help="Check model status")
@click.option("--clear", is_flag=True, help="Clear cached models")
def models(download: bool, status: bool, clear: bool):
    """Manage empathy detection neural networks."""
    from pathlib import Path
    
    cache_dir = Path.home() / ".cache" / "site-empathy" / "models"
    model_files = ["modern_ER.pth", "modern_IP.pth", "modern_EX.pth"]
    
    if clear:
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            console.print(f"[{TEAL}]✓ Neural network cache cleared[/]")
        else:
            console.print(f"[{CORAL}]No cached models to clear[/]")
        return
    
    if status or (not download and not clear):
        console.print(Panel(
            f"[bold]NEURAL NETWORK STATUS[/]\n"
            f"[dim]\"They're either a benefit or a hazard.\"[/]",
            title=f"[{CORAL}]◆ EMPATHY CORES ◆[/]",
            border_style=TEAL
        ))
        console.print()
        console.print(f"[dim]Cache directory:[/] {cache_dir}")
        console.print()
        
        # Check local models
        local_dir = Path(__file__).parent / ".." / ".." / ".." / "models"
        
        table = Table(show_header=True, border_style=TEAL)
        table.add_column("Core", style=f"bold {CORAL}")
        table.add_column("Status")
        table.add_column("Location", style="dim")
        
        for filename in model_files:
            dim = filename.replace("modern_", "").replace(".pth", "")
            
            cache_path = cache_dir / filename
            local_path = local_dir / filename
            
            if cache_path.exists():
                size_mb = cache_path.stat().st_size / 1024 / 1024
                table.add_row(dim, f"[green]● Online ({size_mb:.0f}MB)[/]", str(cache_path))
            elif local_path.exists():
                size_mb = local_path.stat().st_size / 1024 / 1024
                table.add_row(dim, f"[green]● Local ({size_mb:.0f}MB)[/]", str(local_path))
            else:
                table.add_row(dim, f"[{CORAL}]○ Offline[/]", "-")
        
        console.print(table)
        console.print()
        console.print(f"[dim]Run 'site-empathy models --download' to initialize offline cores[/]")
        return
    
    if download:
        console.print(Panel(
            f"[bold]DOWNLOADING EMPATHY CORES[/]\n"
            f"[dim]\"I've seen things you people wouldn't believe...\"[/]",
            title=f"[{CORAL}]◆ NEURAL TRANSFER ◆[/]",
            border_style=TEAL
        ))
        console.print()
        
        try:
            from site_empathy_analysis.models.empathy_model import EmpathyScorer
            
            console.print(f"[{TEAL}]Initializing neural cores (downloading if needed)...[/]")
            scorer = EmpathyScorer()
            console.print()
            console.print(f"[{TEAL}]✓ All empathy cores online![/]")
            
        except Exception as e:
            console.print(f"[{CORAL}]Neural transfer error: {e}[/]")
            console.print()
            console.print(f"[dim]If download failed, you can:[/]")
            console.print(f"  1. Check your network connection")
            console.print(f"  2. Set SITE_EMPATHY_MODEL_URL to a custom source")
            console.print(f"  3. Manually place model files in {cache_dir}")


@main.command()
def info():
    """Learn about the Voight-Kampff empathy framework."""
    console.print(MINI_BANNER)
    
    info_text = f"""
[bold {CORAL}]THE VOIGHT-KAMPFF TEST[/]

[dim]Originally designed by Tyrell Corporation to detect replicants,
this digital version analyzes text for empathic responses.[/]

The framework measures three dimensions of empathy, based on
research by Sharma et al. (2020):

[bold {TEAL}]🔥 EMOTIONAL REACTIONS (ER)[/]
   Expressing warmth, compassion, and concern
   [dim]"I'm so sorry you're going through this"[/]

[bold {TEAL}]🧠 INTERPRETATIONS (IP)[/]
   Acknowledging and naming feelings  
   [dim]"It sounds like you're feeling overwhelmed"[/]

[bold {TEAL}]💬 EXPLORATIONS (EX)[/]
   Asking questions, inviting dialogue
   [dim]"Can you tell me more about what happened?"[/]

[bold]SCORING[/]

Each dimension is scored 0-2:
  [dim]0 = No expression (replicant-like)
  1 = Weak expression (questionable)
  2 = Strong expression (human)[/]

The combined empathy score (0-1) weights the dimensions:
  [{CORAL}]ER: 50%[/] | [{TEAL}]IP: 30%[/] | [dim]EX: 20%[/]

[bold]VERDICT THRESHOLDS[/]

  [green]● HUMAN[/]       ≥ 0.35  High empathy detected
  [yellow]● INCONCLUSIVE[/] ≥ 0.20  Further testing needed
  [{CORAL}]● REPLICANT[/]    < 0.20  Low empathy indicators

[bold]CITATION[/]

  [dim]Sharma, A., Miner, A.S., Atkins, D.C., & Althoff, T. (2020).
  A Computational Approach to Understanding Empathy Expressed in
  Text-Based Mental Health Support. EMNLP 2020.[/]
"""
    
    console.print(Panel(
        info_text, 
        title=f"[{CORAL}]◆ FRAMEWORK DOCUMENTATION ◆[/]",
        border_style=TEAL
    ))
    
    show_quote()


@main.command()
@click.argument("text")
def quick(text: str):
    """Quick empathy test on a text string."""
    try:
        from site_empathy_analysis.models.empathy_model import EmpathyScorer
        
        console.print(f"[{TEAL}]Analyzing empathic response...[/]")
        scorer = EmpathyScorer()
        
        result = scorer.score(text)
        
        # Determine verdict
        score = result.empathy_score
        if score >= 0.35:
            verdict = "[green]HUMAN[/]"
        elif score >= 0.20:
            verdict = "[yellow]INCONCLUSIVE[/]"
        else:
            verdict = f"[{CORAL}]REPLICANT[/]"
        
        # Display results
        table = Table(title=f"[{CORAL}]◆ VOIGHT-KAMPFF RESULT ◆[/]", border_style=TEAL)
        table.add_column("Metric", style=f"bold")
        table.add_column("Value")
        
        table.add_row("Input", f'[dim]"{text[:80]}{"..." if len(text) > 80 else ""}"[/]')
        table.add_row("", "")
        table.add_row("Verdict", verdict)
        table.add_row("Empathy Score", f"[{TEAL}]{score:.3f}[/]")
        table.add_row("", "")
        table.add_row("🔥 Emotional Reaction", f"{result.emotional_reaction:.3f} (level {result.er_level})")
        table.add_row("🧠 Interpretation", f"{result.interpretation:.3f} (level {result.ip_level})")
        table.add_row("💬 Exploration", f"{result.exploration:.3f} (level {result.ex_level})")
        
        if result.empathic_phrases:
            table.add_row("", "")
            table.add_row("Empathic phrases", ", ".join(result.empathic_phrases))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[{CORAL}]Test error: {e}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
