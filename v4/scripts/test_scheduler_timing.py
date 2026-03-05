"""
Unit tests for scheduler timing: next run should be after matchday end (Sunday),
not one day early (Saturday). Uses get_next_run_time() with fixture data.
Run from repo root: uv run python v4/scripts/test_scheduler_timing.py
Or from v4: uv run python scripts/test_scheduler_timing.py
"""

import datetime
import sys
from pathlib import Path

import pandas as pd

# Ensure scripts and v4 are on path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from scheduler import get_latest_match_start_for_matchday, get_next_run_time


def _make_match_row(
    matchday: int,
    season: int,
    dt: datetime.datetime,
    status: str = "future",
) -> dict:
    return {
        "matchDay": matchday,
        "season": season,
        "date": dt,
        "status": status,
    }


def test_get_latest_match_start_for_matchday_returns_sunday():
    """Full matchday with last match Sunday 20:30 -> latest is Sunday 20:30."""
    season = 2025
    matchday = 12
    # Friday 20:30, Saturday 15:30, Saturday 18:30, Sunday 17:30, Sunday 20:30
    dates = [
        datetime.datetime(2025, 3, 7, 20, 30),
        datetime.datetime(2025, 3, 8, 15, 30),
        datetime.datetime(2025, 3, 8, 18, 30),
        datetime.datetime(2025, 3, 9, 17, 30),
        datetime.datetime(2025, 3, 9, 20, 30),
    ]
    rows = [_make_match_row(matchday, season, d) for d in dates]
    match_df = pd.DataFrame(rows)
    result = get_latest_match_start_for_matchday(match_df, matchday, season)
    assert result is not None
    assert result.weekday() == 6  # Sunday
    assert result.hour == 20 and result.minute == 30
    assert result.date() == datetime.date(2025, 3, 9)


def test_get_next_run_time_uses_full_matchday_not_partial():
    """
    next_matchday_df only has Saturday matches (bug case);
    full match_df has Sunday 20:30 as last match.
    Next run must be Sunday 23:30, not Saturday.
    """
    season = 2025
    matchday = 12
    # Full matchday: last match Sunday 20:30
    full_dates = [
        datetime.datetime(2025, 3, 7, 20, 30),
        datetime.datetime(2025, 3, 8, 15, 30),
        datetime.datetime(2025, 3, 8, 18, 30),
        datetime.datetime(2025, 3, 9, 17, 30),
        datetime.datetime(2025, 3, 9, 20, 30),
    ]
    full_rows = [_make_match_row(matchday, season, d) for d in full_dates]
    # Add a future matchday so has_next_matchday is True
    full_rows.append(
        _make_match_row(13, season, datetime.datetime(2025, 3, 16, 20, 30))
    )
    match_df = pd.DataFrame(full_rows)
    # Partial: only Saturday matches (as in the bug: next_matchday_df incomplete)
    saturday_only = [
        datetime.datetime(2025, 3, 8, 15, 30),
        datetime.datetime(2025, 3, 8, 18, 30),
    ]
    next_rows = [_make_match_row(matchday, season, d) for d in saturday_only]
    next_matchday_df = pd.DataFrame(next_rows)
    # Current time: Saturday 22:00, so "next run" is in the future
    current = datetime.datetime(2025, 3, 8, 22, 0)
    next_run = get_next_run_time(match_df, next_matchday_df, current_date=current)
    assert next_run is not None
    # Must be Sunday 23:30 (last match Sunday 20:30 + 3h), not Saturday
    assert next_run.weekday() == 6, "Next run should be Sunday, not Saturday"
    assert next_run.hour == 23 and next_run.minute == 30
    assert next_run.date() == datetime.date(2025, 3, 9)


def test_get_next_run_time_past_schedules_one_hour_from_now():
    """If computed next run is in the past, result is current_date + 1h."""
    season = 2025
    matchday = 12
    # Matchday 12 last match Sunday 20:30; add matchday 13 so has_next_matchday is True
    rows = [
        _make_match_row(matchday, season, datetime.datetime(2025, 3, 9, 20, 30)),
        _make_match_row(13, season, datetime.datetime(2025, 3, 16, 20, 30)),
    ]
    match_df = pd.DataFrame(rows)
    next_matchday_df = pd.DataFrame([rows[0]])
    # Current time: Monday 10:00, so Sunday 23:30 is in the past
    current = datetime.datetime(2025, 3, 10, 10, 0)
    next_run = get_next_run_time(match_df, next_matchday_df, current_date=current)
    assert next_run is not None
    assert next_run == current + datetime.timedelta(hours=1)


if __name__ == "__main__":
    test_get_latest_match_start_for_matchday_returns_sunday()
    print("test_get_latest_match_start_for_matchday_returns_sunday OK")
    test_get_next_run_time_uses_full_matchday_not_partial()
    print("test_get_next_run_time_uses_full_matchday_not_partial OK")
    test_get_next_run_time_past_schedules_one_hour_from_now()
    print("test_get_next_run_time_past_schedules_one_hour_from_now OK")
    print("All scheduler timing tests passed.")
