#!/usr/bin/env python3
"""
Testscript: Scheduler-Logik mit konkreten Daten prüfen.

Szenario:
- Aktueller Spieltag endet am 8.3. (Sonntag)
- Nächster Spieltag endet am 16.3. (Sonntag; Wochenende 15./16.3.)
=> Nächster Job-Lauf soll geplant werden für: 15.3. Abend + 3h = 15.3. 23:30 (So)

Lauf vom Repo-Root: uv run python scripts/check_scheduler_logic.py
"""

import datetime
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from scheduler import (
    get_latest_match_start_for_matchday,
    get_next_run_time,
)

# Konfiguration: Spieltag bis 8.3., nächster bis 15.3. (Sonntag – 16.3.26 wäre Mo!)
CURRENT_MATCHDAY_END = datetime.date(2026, 3, 8)   # 8.3.2026 = Sonntag
NEXT_MATCHDAY_END = datetime.date(2026, 3, 15)    # 15.3.2026 = Sonntag (16.3. = Montag)
SEASON = 2025  # März 2026 → Saison 2025 (get_current_season für März = Vorjahr)
CURRENT_MATCHDAY_NUM = 25
NEXT_MATCHDAY_NUM = 26


def make_row(matchday: int, season: int, dt: datetime.datetime, status: str = "future"):
    return {"matchDay": matchday, "season": season, "date": dt, "status": status}


def build_fixture_full():
    """Voller Spielplan: Spieltag 25 endet 8.3., Spieltag 26 endet 15.3. (So)."""
    y = NEXT_MATCHDAY_END.year
    m = NEXT_MATCHDAY_END.month
    rows = []
    # Spieltag 25 (endet 8.3.) – alle finished
    for d in [
        datetime.datetime(y, 3, 7, 20, 30),
        datetime.datetime(y, 3, 8, 15, 30),
        datetime.datetime(y, 3, 8, 18, 30),
        datetime.datetime(y, 3, 8, 20, 30),
    ]:
        rows.append(make_row(CURRENT_MATCHDAY_NUM, SEASON, d, "finished"))
    # Spieltag 26 (endet 15.3. Sonntag) – alle future, letztes Spiel So 20:30
    for d in [
        datetime.datetime(y, m, 14, 20, 30),
        datetime.datetime(y, m, 15, 15, 30),
        datetime.datetime(y, m, 15, 18, 30),
        datetime.datetime(y, m, 15, 20, 30),
    ]:
        rows.append(make_row(NEXT_MATCHDAY_NUM, SEASON, d, "future"))
    # Spieltag 27 damit has_next_matchday = True
    rows.append(
        make_row(27, SEASON, datetime.datetime(y, m, 22, 20, 30), "future")
    )
    return pd.DataFrame(rows)


def build_next_matchday_full():
    """next_matchday_df wie nach update_next_matchday_df: kompletter nächster Spieltag (26)."""
    y, m = NEXT_MATCHDAY_END.year, NEXT_MATCHDAY_END.month
    rows = []
    for d in [
        datetime.datetime(y, m, 14, 20, 30),
        datetime.datetime(y, m, 15, 15, 30),
        datetime.datetime(y, m, 15, 18, 30),
        datetime.datetime(y, m, 15, 20, 30),
    ]:
        rows.append(make_row(NEXT_MATCHDAY_NUM, SEASON, d, "future"))
    return pd.DataFrame(rows)


def build_next_matchday_only_saturday():
    """Bug-Szenario: next_matchday_df enthält nur Samstag (14.3.), Sonntag 15.3. fehlt."""
    y, m = NEXT_MATCHDAY_END.year, NEXT_MATCHDAY_END.month
    rows = [
        make_row(NEXT_MATCHDAY_NUM, SEASON, datetime.datetime(y, m, 14, 15, 30), "future"),
        make_row(NEXT_MATCHDAY_NUM, SEASON, datetime.datetime(y, m, 14, 20, 30), "future"),
    ]
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Scheduler-Logik Check")
    print(f"  Aktueller Spieltag endet: 8.3.{CURRENT_MATCHDAY_END.year} (So)")
    print(f"  Nächster Spieltag endet:  15.3.{NEXT_MATCHDAY_END.year} (So, 16.3. wäre Mo)")
    print(f"  Erwartung: Nächster Lauf = 15.3.{NEXT_MATCHDAY_END.year} 23:30 (So Abend + 3h)")
    print("=" * 60)

    match_df = build_fixture_full()
    # Simuliere "jetzt" = Sonntag 8.3. 23:00 (nach Ende Spieltag 25)
    now = datetime.datetime(CURRENT_MATCHDAY_END.year, 3, 8, 23, 0)

    # 1) get_latest_match_start_for_matchday für nächsten Spieltag (26)
    latest = get_latest_match_start_for_matchday(
        match_df, NEXT_MATCHDAY_NUM, SEASON
    )
    print(f"\n1) get_latest_match_start_for_matchday(match_df, {NEXT_MATCHDAY_NUM}, {SEASON})")
    if latest:
        wd = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"][latest.weekday()]
        print(f"   => Letztes Spiel: {latest} ({wd})")
        if latest.date() == NEXT_MATCHDAY_END and latest.hour == 20 and latest.minute == 30:
            print(f"   => OK: Sonntag {NEXT_MATCHDAY_END.day}.3. 20:30")
        else:
            print(f"   => FEHLER: Sollte Sonntag {NEXT_MATCHDAY_END.day}.3. 20:30 sein!")
    else:
        print("   => None – FEHLER")

    # 2) get_next_run_time mit vollem next_matchday_df
    next_full = build_next_matchday_full()
    next_run_full = get_next_run_time(match_df, next_full, current_date=now)
    print("\n2) get_next_run_time(match_df, next_matchday_df VOLL, now=8.3. 23:00)")
    if next_run_full:
        wd = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"][next_run_full.weekday()]
        print(f"   => Nächster Lauf: {next_run_full} ({wd})")
        expected = datetime.datetime(NEXT_MATCHDAY_END.year, 3, 15, 23, 30)
        if next_run_full == expected:
            print("   => OK: 15.3. 23:30 (So)")
        else:
            print(f"   => Erwartet: {expected}")

    # 3) Bug-Szenario: next_matchday_df nur mit Samstag
    next_sat = build_next_matchday_only_saturday()
    next_run_sat = get_next_run_time(match_df, next_sat, current_date=now)
    print("\n3) get_next_run_time(match_df, next_matchday_df NUR SAMSTAG, now=8.3. 23:00)")
    print("   (Simuliert: next_matchday_df unvollständig, nur Sa 14.3., So 15.3. fehlt)")
    if next_run_sat:
        wd = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"][next_run_sat.weekday()]
        print(f"   => Nächster Lauf: {next_run_sat} ({wd})")
        if next_run_sat.weekday() == 6 and next_run_sat.date() == NEXT_MATCHDAY_END:
            print("   => OK: Trotzdem Sonntag 15.3. (weil wir match_df nutzen)")
        elif next_run_sat.weekday() == 5:
            print("   => FEHLER: Würde auf Samstag planen (1 Tag zu früh)!")
        else:
            print(f"   => Prüfen: Wochentag {next_run_sat.weekday()}, Datum {next_run_sat.date()}")
    else:
        print("   => None (z.B. has_next_matchday=False)")

    # 4) Kurz: Was liefert .max() nur aus next_matchday_df (alte Logik)?
    print("\n4) Zum Vergleich: next_matchday_df['date'].max() (nur Sa-Subset)")
    only_sat_df = build_next_matchday_only_saturday()
    old_max = only_sat_df["date"].max()
    if hasattr(old_max, "to_pydatetime"):
        old_max = old_max.to_pydatetime()
    print(f"   => {old_max} (nur aus next_matchday_df – könnte zu früh sein)")
    print("   => Mit unserer Logik nutzen wir match_df → 15.3. 20:30 + 3h = 15.3. 23:30")

    print("\n" + "=" * 60)
    if next_run_sat and next_run_sat.weekday() == 6 and next_run_sat.date() == NEXT_MATCHDAY_END:
        print("Ergebnis: Logik OK – nächster Lauf wird auf Sonntag 15.3. geplant.")
    else:
        print("Ergebnis: Bitte Prüfung – ggf. Logik oder Testdaten anpassen.")
    print("=" * 60)


if __name__ == "__main__":
    main()
