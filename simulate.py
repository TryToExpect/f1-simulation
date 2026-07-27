"""
F1 Monte Carlo Race Simulation — standalone, no database required.
Data is fetched directly from the OpenF1 API (openf1.org).

Usage:
  python simulate.py list 2025              # show available races
  python simulate.py run --session 9574     # simulate a race (default: 10,000 iterations)
  python simulate.py run --session 9574 -n 50000 --laps 57 --circuit Monza
"""

import sys
import time
import argparse
import requests
import numpy as np
from collections import defaultdict

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "https://api.openf1.org/v1"

# ---------------------------------------------------------------------------
# Simulation physical parameters
# ---------------------------------------------------------------------------

POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Tyre degradation: delta vs SOFT lap 1, deg/lap, cliff lap
TYRE = {
    "SOFT":         {"delta": 0.0, "deg": 0.12, "cliff": 20, "cliff_mult": 2.5},
    "MEDIUM":       {"delta": 0.5, "deg": 0.07, "cliff": 30, "cliff_mult": 2.0},
    "HARD":         {"delta": 1.0, "deg": 0.04, "cliff": 45, "cliff_mult": 1.8},
    "INTERMEDIATE": {"delta": 5.0, "deg": 0.05, "cliff": 99, "cliff_mult": 1.0},
    "WET":          {"delta": 8.0, "deg": 0.03, "cliff": 99, "cliff_mult": 1.0},
}

# Overtaking-chance coefficient (0 = Monaco, 1 = Monza)
OT_COEFF = {
    "Monaco": 0.08, "Zandvoort": 0.25, "Budapest": 0.20, "Singapore": 0.20,
    "Imola": 0.30, "Suzuka": 0.30, "Barcelona": 0.30, "Silverstone": 0.50,
    "Miami": 0.50, "Jeddah": 0.50, "Abu Dhabi": 0.40, "Melbourne": 0.40,
    "Austin": 0.60, "Shanghai": 0.60, "Montreal": 0.65, "Spa": 0.65,
    "Sao Paulo": 0.65, "Las Vegas": 0.65, "Bahrain": 0.70, "Red Bull Ring": 0.70,
    "Baku": 0.75, "Monza": 0.80, "Lusail": 0.60,
}

# DNF probability per race (2022-2024 history)
DNF_PROB = {
    "Red Bull Racing": 0.04, "McLaren": 0.04, "Mercedes": 0.05,
    "Aston Martin": 0.06,   "Ferrari": 0.07,  "RB": 0.08,
    "Williams": 0.09,       "Alpine": 0.10,   "Kick Sauber": 0.10,
    "Haas F1 Team": 0.11,
}

P_SC_PER_LAP  = 0.035   # Safety Car chance per lap
P_VSC_PER_LAP = 0.025   # Virtual SC chance per lap
SC_DELTA      = 35.0    # seconds slower under SC
VSC_DELTA     = 15.0    # seconds slower under VSC


# ---------------------------------------------------------------------------
# OpenF1 API
# ---------------------------------------------------------------------------

def fetch(endpoint: str, **params) -> list:
    r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Building statistical models from API data
# ---------------------------------------------------------------------------

def build_models(session_key: int) -> dict:
    """
    Fetches session data from the API and builds a dict:
      driver_number -> dict of simulation parameters
    """
    print(f"[1/4] Fetching drivers...")
    drivers_raw = fetch("drivers", session_key=session_key)

    print(f"[2/4] Fetching laps...")
    laps_raw = fetch("laps", session_key=session_key)

    print(f"[3/4] Fetching pit stops...")
    pit_raw = fetch("pit", session_key=session_key)

    print(f"[4/4] Fetching stints...")
    stints_raw = fetch("stints", session_key=session_key)

    # Driver map
    drivers = {d["driver_number"]: d for d in drivers_raw if d.get("driver_number")}

    # Lap times per driver (clean, excludes pit outlap)
    lap_times: dict[int, list] = defaultdict(list)
    for lap in laps_raw:
        dn  = lap.get("driver_number")
        dur = lap.get("lap_duration")
        if dn and dur and 60 < dur < 200 and not lap.get("is_pit_out_lap"):
            lap_times[dn].append(dur)

    # Pit stop durations per team
    team_pits: dict[str, list] = defaultdict(list)
    for pit in pit_raw:
        dn  = pit.get("driver_number")
        dur = pit.get("pit_duration")
        team = drivers.get(dn, {}).get("team_name", "")
        if dur and 1.5 < dur < 60 and team:
            team_pits[team].append(dur)

    # Strategy (stints) per driver
    stints_by: dict[int, list] = defaultdict(list)
    for s in stints_raw:
        dn = s.get("driver_number")
        if dn and s.get("compound") in TYRE:
            stints_by[dn].append({
                "compound":  s["compound"],
                "lap_start": s.get("lap_start", 1),
            })

    # Global fallback (median across all drivers)
    medians = [float(np.median(v)) for v in lap_times.values() if len(v) >= 3]
    global_median = float(np.median(medians)) if medians else 95.0

    models = {}
    for dn, drv in drivers.items():
        team  = drv.get("team_name", "")
        times = lap_times.get(dn, [])

        mean_lt = float(np.median(times)) if len(times) >= 5 else global_median
        std_lt  = max(float(np.std(times)), 0.2) if len(times) >= 5 else 0.5

        pits    = team_pits.get(team, [])
        pit_mean = float(np.mean(pits)) if len(pits) >= 2 else 22.5
        pit_std  = max(float(np.std(pits)), 0.3) if len(pits) >= 2 else 1.5

        strategy = sorted(stints_by.get(dn, []), key=lambda s: s["lap_start"])
        if not strategy:
            strategy = [
                {"compound": "MEDIUM", "lap_start": 1},
                {"compound": "HARD",   "lap_start": 25},
            ]

        models[dn] = {
            "driver_number": dn,
            "acronym":  drv.get("name_acronym", str(dn)),
            "team":     team,
            "mean_lt":  mean_lt,
            "std_lt":   std_lt,
            "pit_mean": pit_mean,
            "pit_std":  pit_std,
            "dnf_prob": DNF_PROB.get(team, 0.08),
            "strategy": strategy,
        }

    print(f"Models ready: {len(models)} drivers\n")
    return models


# ---------------------------------------------------------------------------
# Single race simulation
# ---------------------------------------------------------------------------

def _tyre_delta(compound: str, lap_in_stint: int) -> float:
    cfg = TYRE.get(compound, TYRE["MEDIUM"])
    deg = cfg["delta"] + cfg["deg"] * lap_in_stint
    if lap_in_stint > cfg["cliff"]:
        deg += (lap_in_stint - cfg["cliff"]) * cfg["deg"] * (cfg["cliff_mult"] - 1)
    return deg


def _simulate_once(models: dict, total_laps: int, rng) -> dict:
    """Returns {driver_number: {"position": int, "dnf": bool}}"""
    active      = {dn: True  for dn in models}
    cum_time    = {dn: 0.0   for dn in models}
    laps_done   = {dn: 0     for dn in models}
    stint_lap   = {dn: 1     for dn in models}
    compound    = {}
    pit_sched   = {}

    for dn, m in models.items():
        strat           = m["strategy"]
        compound[dn]    = strat[0]["compound"] if strat else "MEDIUM"
        pit_sched[dn]   = {s["lap_start"] - 1 for s in strat[1:] if s["lap_start"] > 1}

    sc_left    = 0
    vsc_active = False

    for lap in range(1, total_laps + 1):
        # --- Safety Car logic ---
        if sc_left > 0:
            sc_left   -= 1
            sc_on      = True
            vsc_active = False
        elif rng.random() < P_SC_PER_LAP:
            sc_left    = int(rng.integers(3, 8))
            sc_on      = True
            vsc_active = False
        elif rng.random() < P_VSC_PER_LAP:
            sc_left    = int(rng.integers(2, 5))
            sc_on      = False
            vsc_active = True
        else:
            sc_on      = False
            vsc_active = False

        for dn, m in models.items():
            if not active[dn]:
                continue

            # DNF per lap
            if rng.random() < 1 - (1 - m["dnf_prob"]) ** (1 / total_laps):
                active[dn] = False
                continue

            lt = m["mean_lt"] + _tyre_delta(compound[dn], stint_lap[dn])
            lt += rng.normal(0, m["std_lt"])

            if sc_on:
                lt = max(lt, m["mean_lt"] + SC_DELTA)
            elif vsc_active:
                lt = max(lt, m["mean_lt"] + VSC_DELTA)

            cum_time[dn]  += max(lt, 0)
            stint_lap[dn] += 1
            laps_done[dn]  = lap

            # Pit stop
            if lap in pit_sched.get(dn, set()):
                cum_time[dn]  += max(rng.normal(m["pit_mean"], m["pit_std"]), 15.0)
                stint_lap[dn]  = 1
                nxt = [s for s in m["strategy"] if s["lap_start"] > lap]
                compound[dn]   = nxt[0]["compound"] if nxt else "HARD"

    # Final ranking
    finished = sorted(
        [(dn, cum_time[dn]) for dn, ok in active.items() if ok],
        key=lambda x: x[1],
    )
    dnf_list = sorted(
        [(dn, laps_done[dn]) for dn, ok in active.items() if not ok],
        key=lambda x: x[1], reverse=True,
    )

    out = {}
    for pos, (dn, _) in enumerate(finished, 1):
        out[dn] = {"position": pos, "dnf": False}
    for pos, (dn, _) in enumerate(dnf_list, len(finished) + 1):
        out[dn] = {"position": pos, "dnf": True}
    return out


# ---------------------------------------------------------------------------
# Monte Carlo — N simulations
# ---------------------------------------------------------------------------

def run_monte_carlo(models: dict, n: int, total_laps: int, circuit: str) -> list:
    rng = np.random.default_rng(42)

    pos_counts = {dn: np.zeros(22) for dn in models}
    dnf_counts = {dn: 0   for dn in models}
    pos_sums   = {dn: 0   for dn in models}
    pts_sums   = {dn: 0.0 for dn in models}

    print(f"Simulation: {circuit} | {total_laps} laps | N={n:,}")
    t0 = time.perf_counter()

    for i in range(n):
        result = _simulate_once(models, total_laps, rng)
        for dn, res in result.items():
            pos = res["position"]
            if 1 <= pos <= 21:
                pos_counts[dn][pos] += 1
            pos_sums[dn] += pos
            if res["dnf"]:
                dnf_counts[dn] += 1
            else:
                pts_sums[dn] += POINTS.get(pos, 0)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1:>6,}/{n:,}  ({time.perf_counter()-t0:.1f}s)")

    print(f"Done in {time.perf_counter()-t0:.1f}s\n")

    summaries = []
    for dn, m in models.items():
        c = pos_counts[dn]
        summaries.append({
            "acronym":    m["acronym"],
            "team":       m["team"],
            "mean_lt":    round(m["mean_lt"], 3),
            "win_pct":    round(c[1]       / n * 100, 1),
            "podium_pct": round(c[1:4].sum()/ n * 100, 1),
            "top5_pct":   round(c[1:6].sum()/ n * 100, 1),
            "top10_pct":  round(c[1:11].sum()/ n * 100, 1),
            "dnf_pct":    round(dnf_counts[dn] / n * 100, 1),
            "avg_pos":    round(pos_sums[dn] / n, 2),
            "pts_avg":    round(pts_sums[dn] / n, 2),
        })

    summaries.sort(key=lambda x: (-x["win_pct"], -x["podium_pct"]))
    return summaries


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(summaries: list, n: int, session_key: int, circuit: str):
    print(f"{'='*88}")
    print(f"  F1 Monte Carlo | session={session_key} | circuit={circuit} | N={n:,}")
    print(f"{'='*88}")
    print(f"{'#':<4} {'CODE':<6} {'TEAM':<22} {'PACE(s)':<9} {'WIN%':<8} {'PODIUM%':<9} {'TOP10%':<8} {'DNF%':<7} {'AVG':<6} {'PTS/R'}")
    print(f"{'─'*88}")
    for i, s in enumerate(summaries, 1):
        print(
            f"{i:<4} {s['acronym']:<6} {s['team'][:21]:<22} {s['mean_lt']:<9.3f}"
            f"{s['win_pct']:<8.1f} {s['podium_pct']:<9.1f} {s['top10_pct']:<8.1f}"
            f"{s['dnf_pct']:<7.1f} {s['avg_pos']:<6.2f} {s['pts_avg']:.1f}"
        )
    print(f"{'='*88}")


def cmd_list(year: int):
    print(f"Available races {year}:\n")
    meetings = fetch("meetings", year=year)
    print(f"{'SESSION_KEY':<14} {'DATE':<12} {'GRAND PRIX':<38} CIRCUIT")
    print("─" * 80)
    for m in sorted(meetings, key=lambda x: x.get("date_start", "")):
        sessions = fetch("sessions", meeting_key=m["meeting_key"])
        for s in sessions:
            if s.get("session_type") == "Race":
                date = (s.get("date_start") or "")[:10]
                print(
                    f"{s['session_key']:<14} {date:<12} "
                    f"{m.get('meeting_official_name', '')[:37]:<38} "
                    f"{m.get('circuit_short_name', '')}"
                )
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="F1 Monte Carlo Race Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="Show available races")
    p_list.add_argument("year", type=int, nargs="?", default=2025)

    p_run = sub.add_parser("run", help="Run the Monte Carlo simulation")
    p_run.add_argument("--session", type=int, required=True,
                       help="session_key from 'python simulate.py list'")
    p_run.add_argument("--laps",    type=int, default=57,
                       help="Number of laps (default 57)")
    p_run.add_argument("--circuit", type=str, default="default",
                       help="Circuit name — affects the overtaking model")
    p_run.add_argument("-n",        type=int, default=10000,
                       help="Number of simulations (default 10,000)")

    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list(args.year)
    elif args.cmd == "run":
        models  = build_models(args.session)
        results = run_monte_carlo(models, args.n, args.laps, args.circuit)
        print_results(results, args.n, args.session, args.circuit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
