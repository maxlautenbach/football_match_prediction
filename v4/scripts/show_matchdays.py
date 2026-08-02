#!/usr/bin/env python3
"""
Zeigt alle Spiele des aktuellen und nächsten Spieltags aus den lokalen Daten.

Erwartung: Heute = Spieltag 25 (aktuell), nächster = Spieltag 26.
Wenn die Daten andere Spieltage zeigen, Delta-Update / API ausführen.

Lauf von v4 aus: uv run python scripts/show_matchdays.py
"""

# Erwartete Spieltage (Stand: März 2026)
EXPECTED_CURRENT_MATCHDAY = 25
EXPECTED_NEXT_MATCHDAY = 26

import datetime
import pickle
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from data_loader import get_current_season, find_next_match_day


def _format_date(d):
    if hasattr(d, "strftime"):
        return d.strftime("%d.%m.%Y %H:%M")
    return str(d)


def _weekday_de(d):
    if hasattr(d, "weekday"):
        return ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"][d.weekday()]
    return ""


def main():
    now = datetime.datetime.now()
    season = get_current_season(now)
    pickle_file = DATA_DIR / f"match_df_{season}.pck"
    next_path = DATA_DIR / "next_matchday_df.pck"

    print("=" * 70)
    print(f"Datenstand: {now.strftime('%d.%m.%Y %H:%M')}  |  Saison: {season}")
    print(f"Erwartung:  Aktuell = Spieltag {EXPECTED_CURRENT_MATCHDAY}, Nächster = Spieltag {EXPECTED_NEXT_MATCHDAY}")
    print("=" * 70)

    if not pickle_file.exists():
        print(f"\nKeine Daten gefunden: {pickle_file}")
        print("Bitte zuerst Daten laden (z.B. create_datasets oder API).")
        return

    match_df = pd.DataFrame(pickle.load(open(pickle_file, "rb")))
    match_df = match_df[match_df["season"] == season].copy()
    if "date" in match_df.columns and hasattr(match_df["date"].iloc[0], "weekday"):
        pass
    else:
        match_df["date"] = pd.to_datetime(match_df["date"])

    # Aktueller Spieltag = letzter abgeschlossener (alle finished/cancelled)
    matchday_groups = match_df.groupby("matchDay")
    current_matchday_num = None
    for md in sorted(matchday_groups.groups.keys(), reverse=True):
        grp = matchday_groups.get_group(md)
        if (grp["status"] != "future").all():
            current_matchday_num = md
            break

    # Nächster Spieltag aus find_next_match_day (wie im Scheduler)
    next_matchday_df = find_next_match_day(match_df)
    next_matchday_num = int(next_matchday_df["matchDay"].iloc[0]) if len(next_matchday_df) > 0 else None

    # Abweichung von der Erwartung (25/26) anzeigen
    if (current_matchday_num != EXPECTED_CURRENT_MATCHDAY or (next_matchday_num is not None and next_matchday_num != EXPECTED_NEXT_MATCHDAY)):
        print(f"\n⚠️  Daten weichen ab: In den Dateien sind Spieltag {current_matchday_num} (aktuell) und {next_matchday_num} (nächster).")
        print(f"    Erwartet: {EXPECTED_CURRENT_MATCHDAY} / {EXPECTED_NEXT_MATCHDAY}. Bitte Delta-Update ausführen (z.B. predict oder update_match_data_delta).")

    # Optional: gespeichertes next_matchday_df anzeigen
    if next_path.exists():
        saved_next = pd.DataFrame(pickle.load(open(next_path, "rb")))
        if len(saved_next) > 0:
            saved_md = int(saved_next["matchDay"].iloc[0])
            print(f"\nnext_matchday_df.pck enthält: Spieltag {saved_md} ({len(saved_next)} Spiele)")

    # Alle Spiele des aktuellen Spieltags (letzter abgeschlossener)
    print("\n" + "-" * 70)
    if current_matchday_num is not None:
        curr = match_df[match_df["matchDay"] == current_matchday_num].sort_values("date")
        print(f"AKTUELLER SPIELTAG (abgeschlossen): Spieltag {current_matchday_num}  |  {len(curr)} Spiele")
        print("-" * 70)
        for _, row in curr.iterrows():
            d = row["date"]
            if hasattr(d, "to_pydatetime"):
                d = d.to_pydatetime()
            wd = _weekday_de(d) if hasattr(d, "weekday") else ""
            home = row.get("teamHomeName", "")
            away = row.get("teamAwayName", "")
            status = row.get("status", "")
            print(f"  {_format_date(d)}  {wd:2}  |  {home} – {away}  |  {status}")
        if len(curr) > 0:
            last = curr["date"].max()
            if hasattr(last, "to_pydatetime"):
                last = last.to_pydatetime()
            print(f"  → Letztes Spiel: {_format_date(last)} ({_weekday_de(last)})")
    else:
        print("AKTUELLER SPIELTAG: Kein abgeschlossener Spieltag in den Daten.")

    # Alle Spiele des nächsten Spieltags (aus voller match_df)
    print("\n" + "-" * 70)
    if next_matchday_num is not None:
        nxt_full = match_df[match_df["matchDay"] == next_matchday_num].sort_values("date")
        print(f"NÄCHSTER SPIELTAG (aus match_df): Spieltag {next_matchday_num}  |  {len(nxt_full)} Spiele")
        print("-" * 70)
        for _, row in nxt_full.iterrows():
            d = row["date"]
            if hasattr(d, "to_pydatetime"):
                d = d.to_pydatetime()
            wd = _weekday_de(d) if hasattr(d, "weekday") else ""
            home = row.get("teamHomeName", "")
            away = row.get("teamAwayName", "")
            status = row.get("status", "")
            print(f"  {_format_date(d)}  {wd:2}  |  {home} – {away}  |  {status}")
        if len(nxt_full) > 0:
            last = nxt_full["date"].max()
            if hasattr(last, "to_pydatetime"):
                last = last.to_pydatetime()
            print(f"  → Letztes Spiel (für Scheduler): {_format_date(last)} ({_weekday_de(last)}) → Lauf +3h: {_format_date(last + datetime.timedelta(hours=3))}")
    else:
        print("NÄCHSTER SPIELTAG: Keine zukünftigen Spiele in den Daten.")

    # Vergleich: Was steht in next_matchday_df.pck?
    if next_path.exists():
        saved_next = pd.DataFrame(pickle.load(open(next_path, "rb")))
        if len(saved_next) > 0:
            print("\n" + "-" * 70)
            print(f"NÄCHSTER SPIELTAG (nur next_matchday_df.pck): {len(saved_next)} Spiele")
            print("-" * 70)
            for _, row in saved_next.iterrows():
                d = row["date"]
                if hasattr(d, "to_pydatetime"):
                    d = d.to_pydatetime()
                wd = _weekday_de(d) if hasattr(d, "weekday") else ""
                home = row.get("teamHomeName", "")
                away = row.get("teamAwayName", "")
                status = row.get("status", "")
                print(f"  {_format_date(d)}  {wd:2}  |  {home} – {away}  |  {status}")
            last_saved = saved_next["date"].max()
            if hasattr(last_saved, "to_pydatetime"):
                last_saved = last_saved.to_pydatetime()
            print(f"  → max(date) in next_matchday_df: {_format_date(last_saved)} ({_weekday_de(last_saved)})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
