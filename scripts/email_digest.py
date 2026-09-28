"""HTML + plain-text digests for tip and matchday performance emails."""

from __future__ import annotations

import html
from typing import Any, Optional

import pandas as pd


def _fmt_num(value: Any, digits: int = 1) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:.{digits}f}"


def _fmt_z(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:+.2f}"


def _fmt_delta(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:+.1f}"


def _esc(value: Any) -> str:
    return html.escape(str(value) if value is not None else "")


def build_digest_subject(
    *,
    performance: Optional[dict[str, Any]] = None,
    tips_df: Optional[pd.DataFrame] = None,
    startup: bool = False,
) -> str:
    prefix = "Startup · " if startup else ""
    parts: list[str] = []
    if performance and performance.get("n_scored", 0) > 0:
        md = performance["matchday"]
        pts = performance["points"]
        exp = _fmt_num(performance.get("expected"), 1)
        z = _fmt_z(performance.get("z_score"))
        parts.append(f"ST {md}: {pts} Pts (E={exp}, z={z})")
    if tips_df is not None and len(tips_df) > 0:
        tip_md = int(tips_df["Matchday"].iloc[0])
        parts.append(f"Tipps ST {tip_md}")
    if not parts:
        parts.append("Prediction Job")
    return f"[Kicktipp] {prefix}{' · '.join(parts)}"


def build_digest_text(
    *,
    performance: Optional[dict[str, Any]] = None,
    season_stand: Optional[dict[str, Any]] = None,
    tips_df: Optional[pd.DataFrame] = None,
    next_job: Optional[str] = None,
    ops_footer: Optional[str] = None,
) -> str:
    lines: list[str] = []
    lines.append("Football Match Prediction — Digest")
    lines.append("=" * 48)

    if performance and performance.get("n_scored", 0) > 0:
        lines.append("")
        lines.append(
            f"Auswertung Spieltag {performance['matchday']} "
            f"(Saison {performance['season']})"
        )
        lines.append("-" * 48)
        lines.append(
            f"Punkte: {performance['points']}  |  "
            f"Erwartet: {_fmt_num(performance.get('expected'), 1)}  |  "
            f"Δ: {_fmt_delta(performance.get('delta'))}  |  "
            f"z: {_fmt_z(performance.get('z_score'))}"
        )
        lines.append(
            f"Exakt: {performance.get('n_exact', 0)}  |  "
            f"Tendenz: {performance.get('n_outcome', 0)}  |  "
            f"gewertet: {performance.get('n_scored', 0)}/"
            f"{performance.get('n_matches', 0)}"
        )
        lines.append("")
        for m in performance.get("matches", []):
            res = m.get("result") or "—"
            pts = m.get("points")
            pts_s = str(pts) if pts is not None else "—"
            exp_s = _fmt_num(m.get("expected_points"), 2)
            lines.append(
                f"  {m['home_team']} – {m['away_team']}: "
                f"Tipp {m['tip']} → {res}  ({pts_s} Pts, E={exp_s})"
            )

    if season_stand and season_stand.get("n_scored", 0) > 0:
        lines.append("")
        lines.append("Saisonstand (archivierte Tipps)")
        lines.append("-" * 48)
        lines.append(
            f"Punkte: {season_stand['points']}  |  "
            f"E: {_fmt_num(season_stand.get('expected'), 1)}  |  "
            f"Δ: {_fmt_delta(season_stand.get('delta'))}  |  "
            f"z: {_fmt_z(season_stand.get('z_score'))}  |  "
            f"Spieltage: {season_stand.get('n_matchdays', 0)}"
        )

    if tips_df is not None and len(tips_df) > 0:
        tip_md = int(tips_df["Matchday"].iloc[0])
        tip_season = int(tips_df["Season"].iloc[0])
        lines.append("")
        lines.append(f"Tipps Spieltag {tip_md} (Saison {tip_season})")
        lines.append("-" * 48)
        total_exp = 0.0
        n_exp = 0
        for _, row in tips_df.iterrows():
            exp = row.get("Expected_Points", float("nan"))
            try:
                exp_f = float(exp)
            except (TypeError, ValueError):
                exp_f = float("nan")
            exp_s = _fmt_num(exp_f, 2)
            if exp_f == exp_f:
                total_exp += exp_f
                n_exp += 1
            lines.append(
                f"  {row['Home_Team']} – {row['Away_Team']}: "
                f"{row['Prediction']}  (E={exp_s})"
            )
        if n_exp:
            lines.append(f"Σ E[Pts] Spieltag: {total_exp:.1f}")

    if next_job:
        lines.append("")
        lines.append(f"Nächster Job: {next_job}")

    if ops_footer:
        lines.append("")
        lines.append("Ops-Log (Auszug)")
        lines.append("-" * 48)
        # Keep footer short
        footer_lines = ops_footer.strip().splitlines()
        if len(footer_lines) > 40:
            footer_lines = footer_lines[:20] + ["…"] + footer_lines[-15:]
        lines.extend(footer_lines)

    lines.append("")
    return "\n".join(lines)


def build_digest_html(
    *,
    performance: Optional[dict[str, Any]] = None,
    season_stand: Optional[dict[str, Any]] = None,
    tips_df: Optional[pd.DataFrame] = None,
    next_job: Optional[str] = None,
    ops_footer: Optional[str] = None,
) -> str:
    sections: list[str] = []
    sections.append(
        """
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kicktipp Digest</title>
</head>
<body style="margin:0;padding:0;background:#f4f5f7;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1a1a1a;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f5f7;padding:24px 12px;">
<tr><td align="center">
<table role="presentation" width="640" cellpadding="0" cellspacing="0" style="max-width:640px;width:100%;background:#ffffff;border-radius:8px;overflow:hidden;border:1px solid #e5e7eb;">
<tr><td style="padding:20px 24px;background:#0f172a;color:#f8fafc;">
  <div style="font-size:13px;letter-spacing:0.04em;text-transform:uppercase;opacity:0.75;">Kicktipp</div>
  <div style="font-size:22px;font-weight:600;margin-top:4px;">Matchday Digest</div>
</td></tr>
"""
    )

    if performance and performance.get("n_scored", 0) > 0:
        z = performance.get("z_score")
        try:
            z_f = float(z)
            z_ok = z_f == z_f
        except (TypeError, ValueError):
            z_ok = False
            z_f = 0.0
        if z_ok and abs(z_f) >= 2.0:
            badge_bg, badge_fg, badge = "#fef2f2", "#991b1b", "auffällig"
        elif z_ok and abs(z_f) >= 1.0:
            badge_bg, badge_fg, badge = "#fffbeb", "#92400e", "leicht abseits"
        else:
            badge_bg, badge_fg, badge = "#ecfdf5", "#065f46", "im Rahmen"

        sections.append(
            f"""
<tr><td style="padding:20px 24px 8px 24px;">
  <div style="font-size:16px;font-weight:600;">Auswertung Spieltag {_esc(performance['matchday'])}</div>
  <div style="font-size:13px;color:#64748b;margin-top:2px;">Saison {_esc(performance['season'])}</div>
</td></tr>
<tr><td style="padding:8px 24px 16px 24px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f8fafc;border-radius:6px;">
    <tr>
      <td style="padding:14px;text-align:center;width:25%;">
        <div style="font-size:11px;color:#64748b;text-transform:uppercase;">Punkte</div>
        <div style="font-size:22px;font-weight:700;margin-top:4px;">{_esc(performance['points'])}</div>
      </td>
      <td style="padding:14px;text-align:center;width:25%;border-left:1px solid #e2e8f0;">
        <div style="font-size:11px;color:#64748b;text-transform:uppercase;">Erwartet</div>
        <div style="font-size:22px;font-weight:700;margin-top:4px;">{_esc(_fmt_num(performance.get('expected'), 1))}</div>
      </td>
      <td style="padding:14px;text-align:center;width:25%;border-left:1px solid #e2e8f0;">
        <div style="font-size:11px;color:#64748b;text-transform:uppercase;">Δ</div>
        <div style="font-size:22px;font-weight:700;margin-top:4px;">{_esc(_fmt_delta(performance.get('delta')))}</div>
      </td>
      <td style="padding:14px;text-align:center;width:25%;border-left:1px solid #e2e8f0;">
        <div style="font-size:11px;color:#64748b;text-transform:uppercase;">Z</div>
        <div style="font-size:22px;font-weight:700;margin-top:4px;">{_esc(_fmt_z(performance.get('z_score')))}</div>
      </td>
    </tr>
  </table>
  <div style="margin-top:10px;">
    <span style="display:inline-block;padding:4px 10px;border-radius:999px;background:{badge_bg};color:{badge_fg};font-size:12px;font-weight:600;">{_esc(badge)}</span>
    <span style="margin-left:10px;font-size:13px;color:#64748b;">
      Exakt {_esc(performance.get('n_exact', 0))} · Tendenz {_esc(performance.get('n_outcome', 0))} ·
      {_esc(performance.get('n_scored', 0))}/{_esc(performance.get('n_matches', 0))} Spiele
    </span>
  </div>
</td></tr>
<tr><td style="padding:0 24px 20px 24px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:13px;">
    <tr style="background:#f1f5f9;text-align:left;">
      <th style="padding:8px;border-bottom:1px solid #e2e8f0;">Spiel</th>
      <th style="padding:8px;border-bottom:1px solid #e2e8f0;">Tipp</th>
      <th style="padding:8px;border-bottom:1px solid #e2e8f0;">Ergebnis</th>
      <th style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:right;">Pts</th>
      <th style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:right;">E</th>
    </tr>
"""
        )
        for m in performance.get("matches", []):
            pts = m.get("points")
            pts_s = "—" if pts is None else str(pts)
            res = m.get("result") or "—"
            sections.append(
                f"""
    <tr>
      <td style="padding:8px;border-bottom:1px solid #f1f5f9;">{_esc(m['home_team'])} – {_esc(m['away_team'])}</td>
      <td style="padding:8px;border-bottom:1px solid #f1f5f9;font-weight:600;">{_esc(m['tip'])}</td>
      <td style="padding:8px;border-bottom:1px solid #f1f5f9;">{_esc(res)}</td>
      <td style="padding:8px;border-bottom:1px solid #f1f5f9;text-align:right;">{_esc(pts_s)}</td>
      <td style="padding:8px;border-bottom:1px solid #f1f5f9;text-align:right;">{_esc(_fmt_num(m.get('expected_points'), 2))}</td>
    </tr>
"""
            )
        sections.append("</table></td></tr>")

    if season_stand and season_stand.get("n_scored", 0) > 0:
        sections.append(
            f"""
<tr><td style="padding:8px 24px 20px 24px;">
  <div style="font-size:14px;font-weight:600;margin-bottom:8px;">Saisonstand</div>
  <div style="font-size:13px;color:#334155;line-height:1.5;">
    <strong>{_esc(season_stand['points'])}</strong> Pts ·
    E {_esc(_fmt_num(season_stand.get('expected'), 1))} ·
    Δ {_esc(_fmt_delta(season_stand.get('delta')))} ·
    z {_esc(_fmt_z(season_stand.get('z_score')))}
    <span style="color:#64748b;"> · {_esc(season_stand.get('n_matchdays', 0))} Spieltage</span>
  </div>
</td></tr>
"""
        )

    if tips_df is not None and len(tips_df) > 0:
        tip_md = int(tips_df["Matchday"].iloc[0])
        tip_season = int(tips_df["Season"].iloc[0])
        total_exp = 0.0
        n_exp = 0
        tip_rows = []
        for _, row in tips_df.iterrows():
            exp = row.get("Expected_Points", float("nan"))
            try:
                exp_f = float(exp)
            except (TypeError, ValueError):
                exp_f = float("nan")
            if exp_f == exp_f:
                total_exp += exp_f
                n_exp += 1
            tip_rows.append(
                f"""
    <tr>
      <td style="padding:8px;border-bottom:1px solid #f1f5f9;">{_esc(row['Home_Team'])} – {_esc(row['Away_Team'])}</td>
      <td style="padding:8px;border-bottom:1px solid #f1f5f9;font-weight:600;">{_esc(row['Prediction'])}</td>
      <td style="padding:8px;border-bottom:1px solid #f1f5f9;text-align:right;">{_esc(_fmt_num(exp_f, 2))}</td>
    </tr>
"""
            )
        exp_sum = f"{total_exp:.1f}" if n_exp else "n/a"
        sections.append(
            f"""
<tr><td style="padding:8px 24px 8px 24px;">
  <div style="font-size:16px;font-weight:600;">Tipps Spieltag {_esc(tip_md)}</div>
  <div style="font-size:13px;color:#64748b;margin-top:2px;">Saison {_esc(tip_season)} · Σ E[Pts] {_esc(exp_sum)}</div>
</td></tr>
<tr><td style="padding:8px 24px 20px 24px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:13px;">
    <tr style="background:#f1f5f9;text-align:left;">
      <th style="padding:8px;border-bottom:1px solid #e2e8f0;">Spiel</th>
      <th style="padding:8px;border-bottom:1px solid #e2e8f0;">Tipp</th>
      <th style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:right;">E[Pts]</th>
    </tr>
{''.join(tip_rows)}
  </table>
</td></tr>
"""
        )

    if next_job:
        sections.append(
            f"""
<tr><td style="padding:8px 24px 20px 24px;font-size:13px;color:#64748b;">
  Nächster Job: {_esc(next_job)}
</td></tr>
"""
        )

    if ops_footer:
        footer_lines = ops_footer.strip().splitlines()
        if len(footer_lines) > 40:
            footer_lines = footer_lines[:20] + ["…"] + footer_lines[-15:]
        footer_html = _esc("\n".join(footer_lines)).replace("\n", "<br>\n")
        sections.append(
            f"""
<tr><td style="padding:16px 24px 24px 24px;border-top:1px solid #e2e8f0;">
  <div style="font-size:12px;color:#94a3b8;margin-bottom:6px;">Ops-Log (Auszug)</div>
  <div style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:11px;color:#64748b;line-height:1.45;">{footer_html}</div>
</td></tr>
"""
        )

    sections.append(
        """
</table>
</td></tr></table>
</body></html>
"""
    )
    return "".join(sections)
