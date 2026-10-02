from __future__ import annotations
import os
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    FileSizeColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text


_console: Console | None = None
#: The ``sys.stderr`` the cached console was built around, so a harness that
#: swaps the stream out from under us gets a new console instead of writes to a
#: closed file.  See :func:`get_console`.
_console_stream: Any = None


def _encodable(text: str) -> bool:
    """Whether the console's stream can render ``text``.

    Windows consoles still start in a legacy code page unless something sets
    PYTHONUTF8, and writing a glyph the stream cannot encode raises rather than
    degrading.  Probing once lets the ASCII fallbacks below take over instead.
    """
    encoding = getattr(sys.stderr, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


_UNICODE_GLYPHS = _encodable("✓✗•▲")
#: (glyph, style) per severity.
_LEVELS = {
    "info": ("•" if _UNICODE_GLYPHS else "-", "cyan"),
    "success": ("✓" if _UNICODE_GLYPHS else "+", "bold green"),
    "warning": ("▲" if _UNICODE_GLYPHS else "!", "bold yellow"),
    "error": ("✗" if _UNICODE_GLYPHS else "x", "bold red"),
    "debug": ("·" if _UNICODE_GLYPHS else ".", "dim"),
}


def _force_terminal() -> bool | None:
    """Whether to write ANSI escapes even though the stream is not a TTY.

    Hosted notebooks (Colab) pipe the process' stderr into the cell output and
    render escape sequences there, but the pipe fails ``isatty()`` -- so Rich
    strips every colour by default.  ``FORCE_COLOR``/``TTY_COMPATIBLE`` are the
    user's override and Rich reads them itself, so only the Colab case is
    decided here; ``None`` leaves the autodetection alone everywhere else.
    """
    if os.environ.get("FORCE_COLOR") or os.environ.get("TTY_COMPATIBLE"):
        return None
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    return None


def get_console() -> Console:
    """The shared console, rebuilt if ``sys.stderr`` has been replaced.

    ``Console`` captures the stream object at construction, not the name, so a
    console built while some harness had swapped ``sys.stderr`` keeps writing to
    that stream forever -- and once the harness closes it, every later print
    raises ``ValueError: I/O operation on closed file``.  In production the
    stream never changes and this is a no-op; under pytest's capture, or a GUI
    that redirects logs, it is the difference between working and failing
    depending on which code printed first.
    """

    global _console, _console_stream
    stream = sys.stderr
    if _console is None or _console_stream is not stream:
        _console = Console(
            file=stream,
            force_terminal=_force_terminal(),
            highlight=False,
            log_path=False,
            markup=False,
            soft_wrap=True,
        )
        _console_stream = stream
    return _console


def _emit(level: str, message: Any, tag: str | None = None) -> None:
    glyph, style = _LEVELS[level]
    line = Text()
    line.append(f"{glyph} ", style=style)
    if tag:
        line.append(f"{tag} ", style="bold cyan")
    line.append(str(message), style="" if level in ("info", "success") else style)
    get_console().print(line, markup=False, highlight=False)


def info(message: Any, *, tag: str | None = None) -> None:
    """A step that happened as expected."""
    _emit("info", message, tag)


def success(message: Any, *, tag: str | None = None) -> None:
    """A step that finished and produced something."""
    _emit("success", message, tag)


def warning(message: Any, *, tag: str | None = None) -> None:
    """Something worked but not the way the user probably wanted."""
    _emit("warning", message, tag)


class _MetricColumn(ProgressColumn):
    """Render compact task metrics without adding a column when they are empty."""

    def render(self, task):
        metrics = task.fields.get("metrics", "")
        return Text(str(metrics), style="yellow", no_wrap=True)


class _IterationsPerSecondColumn(ProgressColumn):
    """Render the current training throughput."""

    def render(self, task):
        speed = task.speed
        if speed is None or speed <= 0:
            return Text("-- it/s", style="yellow", no_wrap=True)
        return Text(f"{speed:.2f} it/s", style="yellow", no_wrap=True)


def _progress_columns(*, download: bool = False, training: bool = False):
    columns = [
        TextColumn(
            "{task.description}",
            style="bold cyan",
            markup=False,
        ),
        BarColumn(
            complete_style="cyan",
            finished_style="bold green",
            pulse_style="bright_blue",
            bar_width=None,
        ),
        TaskProgressColumn(style="white"),
    ]

    if training:
        columns.extend(
            [
                MofNCompleteColumn(),
                TextColumn("·", style="dim", justify="center"),
                TimeElapsedColumn(),
                TextColumn("·", style="dim", justify="center"),
                TimeRemainingColumn(),
                TextColumn("·", style="dim", justify="center"),
                _IterationsPerSecondColumn(),
                TextColumn("·", style="dim", justify="center"),
                _MetricColumn(),
            ]
        )
    else:
        columns.extend(
            [
                TextColumn("·", style="dim", justify="center"),
                TimeElapsedColumn(),
                TextColumn("·", style="dim", justify="center"),
                TimeRemainingColumn(),
            ]
        )

    if download:
        columns = [
            TextColumn(
                "{task.description}",
                style="bold cyan",
                markup=False,
            ),
            BarColumn(
                complete_style="cyan",
                finished_style="bold green",
                pulse_style="bright_blue",
                bar_width=None,
            ),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("·", style="dim", justify="center"),
            FileSizeColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TextColumn("·", style="dim", justify="center"),
            TimeRemainingColumn(),
        ]

    return columns


#: Seconds between two ``[TASK]`` lines for the same bar.
_TASK_LINE_INTERVAL = 0.5


def reports_to_a_pipe() -> bool:
    """Whether Rich's live bar cannot draw, so progress needs printed lines.
    Asks the console rather than stdout: on Colab stdout is a pipe, but the
    console is forced to a terminal and draws the bar."""
    try:
        return not get_console().is_terminal
    except (AttributeError, ValueError):
        return False


class _ReportingProgress(Progress):
    """A Rich progress that also prints parseable lines when piped.

    Rich renders nothing live when stdout is not a terminal, and these bars
    are transient, so a front-end reading the pipe saw no progress at all for
    preprocessing or extraction.  Each bar now also prints

        [TASK] 120/5166 Slicing & resampling

    at most every :data:`_TASK_LINE_INTERVAL` seconds, plus once when it
    starts and once when it completes.  ``?`` stands for an unknown total.
    The trainer has its own ``[PROGRESS]`` line and does not use this.
    """

    def __init__(self, *args, report: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._report = report
        self._last_report: dict[Any, float] = {}

    def add_task(self, *args, **kwargs):
        task_id = super().add_task(*args, **kwargs)
        self._emit_task_line(task_id, force=True)
        return task_id

    def advance(self, task_id, advance: float = 1) -> None:
        super().advance(task_id, advance)
        self._emit_task_line(task_id)

    def update(self, task_id, *args, **kwargs) -> None:
        super().update(task_id, *args, **kwargs)
        self._emit_task_line(task_id)

    def _emit_task_line(self, task_id, force: bool = False) -> None:
        if not self._report:
            return
        task = self._tasks.get(task_id)
        if task is None:
            return
        total = task.total
        done = task.completed
        finished = total is not None and done >= total
        now = time.monotonic()
        last = self._last_report.get(task_id)
        if finished and last == float("inf"):
            return
        if not (force or finished) and last is not None and now - last < _TASK_LINE_INTERVAL:
            return
        self._last_report[task_id] = float("inf") if finished else now
        shown_total = "?" if total is None else f"{int(total)}"
        description = " ".join(str(task.description).split())
        # Not ``print``: install_rich_print may have replaced it, and Rich
        # would read ``[TASK]`` as markup.
        try:
            sys.stdout.write(f"[TASK] {int(done)}/{shown_total} {description}\n")
            sys.stdout.flush()
        except (OSError, ValueError):
            self._report = False


def create_progress(
    *,
    download: bool = False,
    leave: bool = False,
    training: bool = False,
    disable: bool = False,
) -> Progress:
    return _ReportingProgress(
        *_progress_columns(download=download, training=training),
        report=not disable and not training and reports_to_a_pipe(),
        console=get_console(),
        refresh_per_second=10,
        transient=not leave,
        disable=disable,
        redirect_stdout=False,
        redirect_stderr=False,
    )


@contextmanager
def progress_task(
    total: int | float | None,
    description: str,
    *,
    initial: int | float = 0,
    download: bool = False,
    leave: bool = False,
    training: bool = False,
    disable: bool = False,
):
    progress = create_progress(
        download=download,
        leave=leave,
        training=training,
        disable=disable,
    )
    with progress:
        task_id = progress.add_task(
            description,
            total=total,
            completed=initial,
            metrics="",
        )
        yield progress, task_id


def _parameter_counts(module: Any) -> tuple[int, int]:
    module = getattr(module, "module", module)
    seen: set[int] = set()
    total = 0
    trainable = 0
    for parameter in module.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return total, trainable


def _format_count(value: int) -> str:
    return f"{value:,}"


def print_model_summary(
    models: Iterable[tuple[str, Any]],
    *,
    title: str = "Model Summary",
) -> dict[str, tuple[int, int]]:
    table = Table(
        title=title,
        box=ROUNDED,
        header_style="bold cyan",
        border_style="cyan",
        title_style="bold cyan",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Component", style="cyan")
    table.add_column("Parameters", justify="right")
    table.add_column("Trainable", justify="right", style="green")

    summary: dict[str, tuple[int, int]] = {}
    total_parameters = 0
    total_trainable = 0
    for name, module in models:
        if module is None:
            continue
        total, trainable = _parameter_counts(module)
        summary[name] = (total, trainable)
        total_parameters += total
        total_trainable += trainable
        table.add_row(name, _format_count(total), _format_count(trainable))

        base_module = getattr(module, "module", module)
        for child_name, child in base_module.named_children():
            child_total, child_trainable = _parameter_counts(child)
            table.add_row(
                f"  {child_name}",
                _format_count(child_total),
                _format_count(child_trainable),
            )

    table.add_section()
    table.add_row(
        "Total",
        _format_count(total_parameters),
        _format_count(total_trainable),
        style="bold",
    )
    get_console().print(table)
    return summary


def print_settings_panel(
    rows: Iterable[tuple[str, Any]],
    *,
    title: str | None = None,
) -> None:
    table = Table.grid(padding=(0, 1), expand=False)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white", overflow="fold")
    for label, value in rows:
        table.add_row(str(label), str(value))

    get_console().print(
        Panel(
            table,
            title=title,
            title_align="left",
            border_style="cyan",
            box=ROUNDED,
            expand=False,
            padding=(0, 1),
        )
    )
