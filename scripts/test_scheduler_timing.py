"""
Unit tests for scheduler timing: next run should be after matchday end (Sunday),
not one day early (Saturday). Uses get_next_run_time() with fixture data.
Run from repo root: uv run python scripts/test_scheduler_timing.py
"""

import datetime
import sys
from pathlib import Path

import pandas as pd

# Ensure scripts and project root are on path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from scheduler import (
    defer_until_after_in_progress_matchday,
    get_latest_match_start_for_matchday,
    get_next_run_time,
    matchday_in_progress,
)


def _make_match_row(
    matchday: int,
    season: int,
    dt: datetime.datetime,
    status: str = "future",
    league: str = "bl1",
) -> dict:
    return {
        "matchDay": matchday,
        "season": season,
        "date": dt,
        "status": status,
        "league": league,
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


def test_bl1_latest_ignores_later_bl2_same_matchday():
    """Kicktipp scheduling must use BL1 end, not a later BL2 kickoff."""
    season = 2026
    matchday = 5
    rows = [
        _make_match_row(
            matchday, season, datetime.datetime(2026, 10, 10, 15, 30), league="bl1"
        ),
        _make_match_row(
            matchday, season, datetime.datetime(2026, 10, 11, 17, 30), league="bl1"
        ),
        _make_match_row(
            matchday, season, datetime.datetime(2026, 10, 12, 20, 30), league="bl2"
        ),
    ]
    latest = get_latest_match_start_for_matchday(pd.DataFrame(rows), matchday, season)
    assert latest == datetime.datetime(2026, 10, 11, 17, 30)


def test_defer_saturday_partial_matchday():
    """Saturday 18:30 with Sunday games left → defer to Sunday 20:30, no tip."""
    season = 2026
    matchday = 5
    rows = [
        _make_match_row(
            matchday,
            season,
            datetime.datetime(2026, 10, 9, 20, 30),
            status="finished",
        ),
        _make_match_row(
            matchday,
            season,
            datetime.datetime(2026, 10, 10, 15, 30),
            status="finished",
        ),
        _make_match_row(
            matchday,
            season,
            datetime.datetime(2026, 10, 11, 15, 30),
            status="future",
        ),
        _make_match_row(
            matchday,
            season,
            datetime.datetime(2026, 10, 11, 17, 30),
            status="future",
        ),
        _make_match_row(6, season, datetime.datetime(2026, 10, 18, 15, 30)),
    ]
    match_df = pd.DataFrame(rows)
    assert matchday_in_progress(match_df, matchday, season)
    next_partial = match_df[
        (match_df["matchDay"] == matchday) & (match_df["status"] == "future")
    ]
    current = datetime.datetime(2026, 10, 10, 18, 30)
    defer_to = defer_until_after_in_progress_matchday(
        match_df, next_partial, current_date=current
    )
    assert defer_to == datetime.datetime(2026, 10, 11, 20, 30)


def test_no_defer_after_matchday_end_buffer():
    """After last kickoff + 3h, in-progress leftover must not defer forever."""
    season = 2026
    matchday = 5
    rows = [
        _make_match_row(
            matchday,
            season,
            datetime.datetime(2026, 10, 11, 17, 30),
            status="future",
        ),
        _make_match_row(
            matchday,
            season,
            datetime.datetime(2026, 10, 10, 15, 30),
            status="finished",
        ),
    ]
    match_df = pd.DataFrame(rows)
    next_partial = match_df[match_df["status"] == "future"]
    current = datetime.datetime(2026, 10, 11, 20, 30)
    assert (
        defer_until_after_in_progress_matchday(
            match_df, next_partial, current_date=current
        )
        is None
    )


if __name__ == "__main__":
    test_get_latest_match_start_for_matchday_returns_sunday()
    print("test_get_latest_match_start_for_matchday_returns_sunday OK")
    test_get_next_run_time_uses_full_matchday_not_partial()
    print("test_get_next_run_time_uses_full_matchday_not_partial OK")
    test_get_next_run_time_past_schedules_one_hour_from_now()
    print("test_get_next_run_time_past_schedules_one_hour_from_now OK")
    test_bl1_latest_ignores_later_bl2_same_matchday()
    print("test_bl1_latest_ignores_later_bl2_same_matchday OK")
    test_defer_saturday_partial_matchday()
    print("test_defer_saturday_partial_matchday OK")
    test_no_defer_after_matchday_end_buffer()
    print("test_no_defer_after_matchday_end_buffer OK")
    print("All scheduler timing tests passed.")
