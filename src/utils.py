"""
Cobblestone Power — shared utilities.

Purpose:
    Logging setup, retry helpers, parquet I/O, timezone helpers, and
    common DataFrame validation used across the pipeline.

Inputs:
    Paths, DataFrames, callables for retry wrapping.

Outputs:
    Configured loggers, saved/loaded parquet files, validated DataFrames.

Side Effects:
    Writes log files under outputs/logs/; creates parent directories on save.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd

from config.settings import FIGURE_DPI, RANDOM_SEED, get_settings

T = TypeVar("T")

# Retry defaults
MAX_RETRIES: int = 5
BASE_DELAY_SECONDS: float = 2.0
LEAP_YEAR_HOURS: int = 8784
NON_LEAP_YEAR_HOURS: int = 8760


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Configure root logging with file + console handlers.

    Args:
        level: Optional log level override (default from settings).

    Returns:
        Root logger for the cobblestone pipeline.

    Example:
        >>> logger = setup_logging("INFO")
        >>> logger.info("pipeline start")
    """
    settings = get_settings()
    log_level = (level or settings.log_level).upper()
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    log_file = settings.logs / "pipeline_run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(log_format))
        root.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(log_format))
        root.addHandler(sh)

    # Fix seeds in one place for reproducibility
    np.random.seed(RANDOM_SEED)

    return logging.getLogger("cobblestone")


def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY_SECONDS,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator: retry a function with exponential backoff.

    Args:
        max_retries: Maximum attempts (default 5).
        base_delay: Initial delay in seconds (doubles each retry).
        exceptions: Exception types that trigger a retry.

    Returns:
        Decorated callable.

    Raises:
        Last exception if all retries fail.

    Example:
        >>> @retry_with_backoff(max_retries=3)
        ... def fetch():
        ...     return 1
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    delay = base_delay * (2 ** (attempt - 1))
                    logging.getLogger(fn.__module__).warning(
                        "Attempt %s/%s failed for %s: %s — retrying in %.1fs",
                        attempt,
                        max_retries,
                        fn.__name__,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


def ensure_utc_index(obj: Union[pd.Series, pd.DataFrame], name: str = "index") -> Union[pd.Series, pd.DataFrame]:
    """
    Ensure a DatetimeIndex is timezone-aware UTC.

    Args:
        obj: Series or DataFrame with DatetimeIndex.
        name: Label used in error messages.

    Returns:
        Same object with UTC DatetimeIndex.

    Raises:
        TypeError: If index is not datetime-like.
    """
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex, got {type(obj.index)}")
    if obj.index.tz is None:
        obj = obj.copy()
        obj.index = obj.index.tz_localize("UTC")
    else:
        obj = obj.copy()
        obj.index = obj.index.tz_convert("UTC")
    return obj


def hours_in_year(year: int) -> int:
    """
    Return expected hourly row count for a calendar year.

    Args:
        year: Gregorian year.

    Returns:
        8784 for leap years, else 8760.
    """
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return LEAP_YEAR_HOURS if is_leap else NON_LEAP_YEAR_HOURS


def assert_hourly_completeness(index: pd.DatetimeIndex, year: int) -> None:
    """
    Assert an index has exactly the expected number of hours for a year.

    Args:
        index: UTC DatetimeIndex filtered to `year`.
        year: Calendar year to check.

    Raises:
        AssertionError: If row count mismatches leap/non-leap expectation.
    """
    year_index = index[index.year == year]
    expected = hours_in_year(year)
    actual = len(year_index)
    assert actual == expected, (
        f"Year {year}: expected {expected} hourly rows, got {actual}"
    )


def validate_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    context: str = "DataFrame",
) -> None:
    """
    Validate that required columns exist.

    Args:
        df: Input DataFrame.
        required: Column names that must be present.
        context: Label for error messages.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def save_parquet(obj: Union[pd.Series, pd.DataFrame], path: Path) -> Path:
    """
    Save a Series/DataFrame to parquet, creating parent dirs.

    Args:
        obj: Data to persist.
        path: Destination .parquet path.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.Series):
        obj.to_frame(name=obj.name or "value").to_parquet(path)
    else:
        obj.to_parquet(path)
    logging.getLogger(__name__).info("Saved parquet → %s (%s rows)", path, len(obj))
    return path


def load_parquet(path: Path, as_series: bool = False, series_col: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
    """
    Load a parquet file as DataFrame or Series.

    Args:
        path: Source .parquet path.
        as_series: If True, return first (or named) column as Series.
        series_col: Optional column name when as_series=True.

    Returns:
        Loaded Series or DataFrame with UTC index when possible.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = ensure_utc_index(df)  # type: ignore[assignment]
    if as_series:
        col = series_col or df.columns[0]
        return ensure_utc_index(df[col])  # type: ignore[return-value]
    return df


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """
    Append a JSON record as one line to a JSONL log file.

    Args:
        path: Destination .jsonl path.
        record: JSON-serialisable dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def utc_now_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def save_figure(fig: Any, path: Path, dpi: int = FIGURE_DPI) -> Path:
    """
    Save a matplotlib figure at production DPI.

    Args:
        fig: Matplotlib Figure.
        path: Destination path (.png).
        dpi: Resolution (default 300).

    Returns:
        Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logging.getLogger(__name__).info("Saved figure → %s", path)
    return path


def safe_eval_condition(condition: str, df: pd.DataFrame) -> pd.Series:
    """
    Safely evaluate a boolean pandas expression for QA rules.

    Args:
        condition: Python expression using `df`, `pd`, `np`.
        df: DataFrame in scope as `df`.

    Returns:
        Boolean Series of violations (True = violation).

    Raises:
        Exception: Propagates evaluation errors to caller for logging.
    """
    # Restricted globals — no builtins beyond what's needed
    safe_globals: Dict[str, Any] = {"df": df, "pd": pd, "np": np, "__builtins__": {}}
    result = eval(condition, safe_globals, {})  # noqa: S307 — intentional controlled eval
    if isinstance(result, pd.Series):
        return result.fillna(False).astype(bool)
    if isinstance(result, np.ndarray):
        return pd.Series(result.astype(bool), index=df.index)
    if isinstance(result, (list, tuple)):
        return pd.Series(np.asarray(result, dtype=bool), index=df.index)
    return pd.Series(bool(result), index=df.index)


def cyclic_encode(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cyclically encode a periodic feature as sin/cos.

    Args:
        values: Numeric values (e.g. hour 0-23).
        period: Cycle length (e.g. 24 for hours).

    Returns:
        (sin_component, cos_component) arrays.
    """
    angle = 2.0 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """
    Write a JSON file with UTF-8 encoding and indentation.

    Args:
        path: Destination path.
        payload: JSON-serialisable dict.

    Returns:
        Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    """
    Read a JSON file.

    Args:
        path: Source path.

    Returns:
        Parsed dict.

    Raises:
        FileNotFoundError: If path missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_required_feature_groups() -> List[str]:
    """Return canonical feature group labels A–H."""
    return [
        "A_fundamental",
        "B_fuel_carbon",
        "C_calendar",
        "D_lags",
        "E_rolling",
        "F_interactions",
        "G_regime",
        "H_structural",
    ]
