"""
Logging utilities.

Provides unified logging functionality.
"""
import logging
from pathlib import Path
from typing import Any, Optional
from zlib import crc32
from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    name: str = "evaluation"
) -> logging.Logger:
    """
    Setup logger.
    
    Args:
        log_file: Log file path (optional)
        level: Log level
        name: Logger name
        
    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add Rich Console Handler (colored output)
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=False,
        show_path=False
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # Add file Handler (if log file is specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_console() -> Console:
    """Get Rich Console instance."""
    return Console()


class RunLogger:
    """Run-scoped console and file logger."""

    def __init__(self, logger: logging.Logger, console: Console):
        self.logger = logger
        self.console = console

    def _console(self, message: str, *, style: Optional[str] = None) -> None:
        if style:
            self.console.print(message, style=style)
        else:
            self.console.print(message)

    def event(self, message: str, *args: Any, style: Optional[str] = None) -> None:
        if args:
            message = message % args
        self._console(message, style=style)
        self.logger.info(message)

    def warning(self, message: str, *args: Any) -> None:
        if args:
            message = message % args
        self._console(f"[yellow]{message}[/yellow]")
        self.logger.warning(message)

    def error(self, message: str, *args: Any) -> None:
        if args:
            message = message % args
        self._console(f"[red]{message}[/red]")
        self.logger.error(message)

    def stage_start(self, stage_number: int, stage_name: str) -> None:
        self.event(
            f"Stage {stage_number}: {stage_name} started",
            style="bold cyan",
        )

    def stage_skip(self, stage_number: int, stage_name: str, reason: str) -> None:
        self.event(
            f"Stage {stage_number}: {stage_name} skipped - {reason}",
            style="yellow",
        )

    def stage_complete(self, stage_number: int, stage_name: str) -> None:
        self.event(
            f"Stage {stage_number}: {stage_name} completed",
            style="green",
        )

    def artifact_written(
        self,
        filename: str,
        *,
        rows: Optional[int] = None,
        reason: str = "",
    ) -> None:
        parts = [f"{filename} written"]
        if rows is not None:
            parts.append(f"rows={rows}")
        if reason:
            parts.append(f"reason={reason}")
        self.event(" ".join(parts))


def setup_run_logger(
    log_file: Path,
    run_id: str,
    level: int = logging.INFO,
) -> RunLogger:
    """Set up a run-scoped logger that writes only to this run's log file."""
    console = get_console()
    safe_run_id = str(run_id or "unknown").replace(".", "_")
    path_hash = f"{crc32(str(log_file.resolve()).encode('utf-8')):08x}"
    logger = logging.getLogger(f"subtlememory.run.{safe_run_id}.{path_hash}")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return RunLogger(logger=logger, console=console)
