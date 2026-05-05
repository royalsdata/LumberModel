"""
lumber_plus_update.py
=====================
Fetches current-season Statcast data directly from Baseball Savant,
computes Lumber+, Swing+, Damage+, and Plate+, then regenerates
the public leaderboard HTML file.

Usage:
    python lumber_plus_update.py              # defaults to current year
    python lumber_plus_update.py --year 2025  # specific season
    python lumber_plus_update.py --min-pa 100 # override PA floor

Requirements:
    pip install pandas requests
"""

import argparse
import io
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

# ── CONFIG ────────────────────────────────────────────────────────────────────

CURRENT_YEAR = date.today().year
OUTPUT_HTML  = Path("lumber_plus_leaderboard.html")
MIN_PA       = 50

SAVANT_URL = (
    "https://baseballsavant.mlb.com/leaderboard/custom"
    "?year={year}"
    "&type=batter"
    "&filter="
    "&min={min_pa}"
    "&selections=pa"
    "%2Ck_percent%2Cbb_percent"
    "%2Con_base_plus_slg%2Cwoba"
    "%2Cavg_swing_speed%2Cfast_swing_rate%2Csquared_up_swing%2Cideal_angle_rate"
    "%2Cexit_velocity_avg%2Csweet_spot_percent%2Cbarrel_batted_rate%2Chard_hit_percent"
    "%2Cz_swing_percent%2Coz_swing_percent%2Cmeatball_swing_percent"
    "%2Ciz_contact_percent%2Cwhiff_percent%2Cpull_percent"
    "&chart=false&x=pa&y=pa&r=no&chartType=beeswarm"
    "&sort=xwoba&sortDir=desc"
    "&csv=true"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ── MODEL WEIGHTS ─────────────────────────────────────────────────────────────

SWING_WEIGHTS = {
    "avg_swing_speed":  0.35,
    "fast_swing_rate":  0.30,
    "squared_up_swing": 0.25,
    "ideal_angle_rate": 0.10,
}

DAMAGE_WEIGHTS = {
    "barrel_batted_rate": 0.30,
    "exit_velocity_avg":  0.25,
    "sweet_spot_percent": 0.175,
    "hard_hit_percent":   0.175,
    "pull_percent":       0.10,
}

PLATE_WEIGHTS = {
    "oz_swing_percent":       -0.35,
    "meatball_swing_percent":  0.30,
    "bb_percent":              0.20,
    "k_percent":              -0.15,
}

LUMBER_WEIGHTS = {"Swing+": 0.35, "Damage+": 0.40, "Plate+": 0.25}


# ── FETCH ─────────────────────────────────────────────────────────────────────

def fetch_savant(year: int, min_pa: int) -> pd.DataFrame:
    url = SAVANT_URL.format(year=year, min_pa=min_pa)
    print(f"  Fetching Savant data for {year} (min {min_pa} PA)...")

    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            if resp.text.strip().startswith("<"):
                raise ValueError("Savant returned HTML instead of CSV — likely rate-limited.")
            df = pd.read_csv(io.StringIO(resp.text))
            print(f"  ✓ {len(df)} players, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt == 3:
                sys.exit("ERROR: All fetch attempts failed. Try again later.")
            import time; time.sleep(10)


# ── MODEL ─────────────────────────────────────────────────────────────────────

def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    return (series - series.mean()) / std if std > 0 else pd.Series(0.0, index=series.index)


def compute_component(df: pd.DataFrame, weights: dict, name: str) -> pd.Series:
    composite = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col not in df.columns:
            print(f"  WARNING: '{col}' missing — skipping from {name}")
            continue
        composite += zscore(df[col]) * w
    return (100 + composite * 10).round(1)


def compute_lumber_plus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalise name: "Last, First" -> "First Last"
    name_col = next((c for c in ["last_name, first_name", "player_name"] if c in df.columns), None)
    if not name_col:
        sys.exit("ERROR: no name column found in Savant CSV.")
    if name_col == "last_name, first_name":
        df["name"] = df[name_col].apply(
            lambda s: f"{s.split(', ')[1]} {s.split(', ')[0]}"
            if isinstance(s, str) and ", " in s else str(s)
        )
    else:
        df["name"] = df[name_col]

    # Convert 0-1 decimals to percentages if needed
    pct_cols = [
        "k_percent", "bb_percent", "fast_swing_rate", "squared_up_swing",
        "ideal_angle_rate", "sweet_spot_percent", "barrel_batted_rate",
        "hard_hit_percent", "pull_percent", "oz_swing_percent",
        "meatball_swing_percent", "iz_contact_percent", "whiff_percent",
        "z_swing_percent",
    ]
    for col in pct_cols:
        if col in df.columns and df[col].dropna().max() <= 1.05:
            df[col] = (df[col] * 100).round(1)

    df["Swing+"]  = compute_component(df, SWING_WEIGHTS,  "Swing+")
    df["Damage+"] = compute_component(df, DAMAGE_WEIGHTS, "Damage+")
    df["Plate+"]  = compute_component(df, PLATE_WEIGHTS,  "Plate+")
    df["Lumber+"] = (
        df["Swing+"]  * LUMBER_WEIGHTS["Swing+"]  +
        df["Damage+"] * LUMBER_WEIGHTS["Damage+"] +
        df["Plate+"]  * LUMBER_WEIGHTS["Plate+"]
    ).round(1)

    print(f"  Lumber+ mean={df['Lumber+'].mean():.1f}  SD={df['Lumber+'].std():.1f}")
    return df


# ── HTML INJECTION ────────────────────────────────────────────────────────────

def update_html(df: pd.DataFrame, year: int) -> None:
    if not OUTPUT_HTML.exists():
        sys.exit(f"ERROR: {OUTPUT_HTML} not found. Place this script in the same folder.")

    players = (
        df[["name", "pa", "Lumber+", "Swing+", "Damage+", "Plate+"]]
        .rename(columns={"Lumber+": "lumber", "Swing+": "swing",
                          "Damage+": "damage", "Plate+": "plate"})
        .assign(pa=lambda d: d["pa"].astype(int))
        .sort_values("lumber", ascending=False)
        .to_dict("records")
    )

    data_json = json.dumps(players, separators=(",", ":"), ensure_ascii=False)
    today_str = date.today().strftime("%-m/%-d/%y")

    html = OUTPUT_HTML.read_text(encoding="utf-8")

    var_name = "DATA_2026" if year == 2026 else "DATA_2025"
    html = re.sub(
        rf"const {var_name} = \[.*?\];",
        f"const {var_name} = {data_json};",
        html, flags=re.DOTALL
    )

    html = re.sub(r"Last Updated:.*?(?=<)", f"Last Updated: {today_str}", html)

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"  ✓ {len(players)} players written to {OUTPUT_HTML}")
    print(f"  ✓ Last Updated → {today_str}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Update Lumber+ leaderboard")
    parser.add_argument("--year",   type=int, default=CURRENT_YEAR)
    parser.add_argument("--min-pa", type=int, default=MIN_PA)
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Lumber+ Update — {args.year}")
    print(f"{'='*50}")

    df = fetch_savant(args.year, args.min_pa)
    df = compute_lumber_plus(df)

    print(f"\nTop 10:")
    top = df.nlargest(10, "Lumber+")[["name", "pa", "Lumber+", "Swing+", "Damage+", "Plate+"]]
    print(top.to_string(index=False))

    update_html(df, args.year)
    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    main()
