# F1 Monte Carlo Race Simulator by F1 Enjoyer :D

Welcome to my ongoing project on probabilistic modelling of Formula 1 races.

This simulator combines real-world telemetry (via the OpenF1 API) with statistical
modelling and Monte Carlo simulation to estimate race outcome probabilities —
win/podium/points chances, tyre strategy effects, and the impact of safety cars
and DNFs on final standings.

The goal is to bridge probability theory with practical software engineering:
data is used to calibrate distributions (lap time variance, pit stop duration,
DNF rates), which then feed thousands of simulated race realizations.

⚠️ Early-stage / work in progress.

All data comes live from the [OpenF1 API](https://openf1.org) — no database, no API key, no setup beyond `pip install`.

## Features

- Monte Carlo race simulation (default 10,000 runs) driven by real lap times, pit stop durations, and tyre strategies pulled from OpenF1
- Per-compound tyre degradation model with a performance "cliff" once a tyre is overworked
- Random Safety Car / Virtual Safety Car periods, each lap
- Per-lap DNF risk derived from team reliability
- `predict` mode: simulate a future or hypothetical race using the current grid and recent-race statistics, with no session data required
- Zero configuration — just Python and two libraries

## How it works

### `run` — simulate a race that already happened

For a given `session_key`, `build_models()` pulls from OpenF1:

- **Drivers** — number, team, acronym
- **Laps** — used to compute each driver's median race pace and its variance (pit out-laps and outlier laps outside 60–200s are excluded)
- **Pit stops** — used to compute each team's average pit stop duration
- **Stints** — used to reconstruct each driver's actual tyre strategy for that race

If a driver doesn't have enough clean laps (or pit data), the model falls back to the field median / a generic default so the simulation still runs.

### `predict` — simulate a race with no session data

`run` only works for races that have already been run — OpenF1 has nothing to fetch for a session that hasn't happened. `predict` works around this by building a model from history instead of a single session:

1. Takes the **current grid/team lineup** from the most recently completed race.
2. Aggregates each driver's **pace relative to the field**, DNF rate, and pit stop times across the last `--history` races (default 40), weighting more recent races higher (exponential decay).
3. Takes the **baseline lap time** from the most recent past race actually held at `--circuit`, and adds each driver's relative pace delta on top of it.
4. Feeds the resulting per-driver models into the same Monte Carlo engine as `run`.

This means `predict` answers "how would the current grid likely perform at this circuit," blending long-term form with track-specific pace — it is a statistical estimate, not a strategy-aware forecast (see [Limitations](#limitations)).

### The race model (used by both `run` and `predict`)

Each simulated race runs lap-by-lap for every driver:

- **Tyre degradation** — lap time increases with tyre age per compound (`SOFT`/`MEDIUM`/`HARD`/`INTERMEDIATE`/`WET`), accelerating sharply past that compound's "cliff" lap
- **Pace variance** — each lap's time is the driver's mean pace plus tyre degradation plus Gaussian noise (`std_lt`)
- **Safety Car / Virtual Safety Car** — each lap has an independent chance of triggering a multi-lap SC or VSC period, during which every driver is capped to a slower minimum lap time
- **DNF** — each lap carries a small retirement probability derived from the driver's/team's overall reliability
- **Pit stops** — happen at the strategy's scheduled laps, costing time drawn from that team's pit stop distribution, after which the tyre compound switches per the strategy

After `N` runs, results are aggregated into win / podium / top-10 / DNF percentages, average finishing position, and average points per race.

## Requirements

- Python 3.10+
- `requests`
- `numpy`

Install:

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Find a session_key for a race that has already happened
python simulate.py list 2025

# 2. Simulate it
python simulate.py run --session 9912

# ...or predict a race with no session data at all
python simulate.py predict --circuit Zandvoort
```

## Commands

### `list [year]`

Lists every race session for a given season (default `2025`) with its `session_key`, date, Grand Prix name, and circuit.

```bash
python simulate.py list 2025
```

Only sessions that have already taken place have usable data in OpenF1 — use this to find a valid `session_key` for `run`.

### `run --session SESSION_KEY [options]`

Simulates a specific, already-completed race session.

| Flag | Default | Description |
|------|---------|-------------|
| `--session` | *required* | `session_key` from `list` |
| `--laps` | `57` | Number of race laps |
| `--circuit` | `default` | Label shown in the results header (see [Limitations](#limitations) — it does not currently change the simulation itself) |
| `-n` | `10000` | Number of Monte Carlo iterations |

```bash
python simulate.py run --session 9574 -n 50000 --laps 57 --circuit Monza
```

### `predict --circuit CIRCUIT [options]`

Simulates a race with **no session data required** — useful for a race that hasn't happened yet, or any what-if scenario for the current grid.

| Flag | Default | Description |
|------|---------|-------------|
| `--circuit` | *required* | Circuit name, must match OpenF1's `circuit_short_name` exactly, case-insensitively (e.g. `Zandvoort`, `Monza`, `Spa`) |
| `--laps` | auto | Number of laps; auto-detected from the most recent past race at that circuit if omitted |
| `--history` | `40` | Number of past races used to build driver models (more = more stable, less reactive to current form) |
| `-n` | `10000` | Number of Monte Carlo iterations |

```bash
python simulate.py predict --circuit Monza --history 15 -n 5000
```

`--history 40` (the default) fetches roughly 40 sessions' worth of data and can take a few minutes — OpenF1 rate-limits aggressively, and the script automatically retries with backoff on `429` responses. For a much faster (if slightly less stable) result, lower `--history` to around 15–20.

## Reading the output

```
========================================================================================
  F1 Monte Carlo | session=9912 | circuit=default | N=10,000
========================================================================================
#    CODE   TEAM                   PACE(s)   WIN%     PODIUM%   TOP10%   DNF%    AVG    PTS/R
────────────────────────────────────────────────────────────────────────────────────────
1    VER    Red Bull Racing        82.890   62.8     92.1      96.0    4.0     2.23   21.2
2    LEC    Ferrari                83.132   19.7     72.8      93.2    6.8     3.70   16.1
...
========================================================================================
```

| Column | Meaning |
|--------|---------|
| `PACE(s)` | Driver's modeled mean lap time (median race pace for `run`; baseline + relative delta for `predict`) |
| `WIN%` | % of simulations finished in P1 |
| `PODIUM%` | % of simulations finished P1–P3 |
| `TOP10%` | % of simulations finished P1–P10 (i.e. scoring, ignoring DNF) |
| `DNF%` | % of simulations that didn't finish |
| `AVG` | Average finishing position across all simulations (DNFs count as a low finishing position) |
| `PTS/R` | Average championship points scored per race |

Rows are sorted by `WIN%`, then `PODIUM%`. Note: a top-5 percentage is also computed internally but isn't currently printed in the table.

## Tuning the model

Constants live at the top of `simulate.py`:

- `TYRE` — per-compound degradation rate, cliff lap, and cliff severity multiplier
- `DNF_PROB` — per-team reliability fallback used when a driver doesn't have enough race history (`run`) or race starts (`predict`)
- `P_SC_PER_LAP` / `P_VSC_PER_LAP` — Safety Car / Virtual Safety Car chance per lap
- `SC_DELTA` / `VSC_DELTA` — how many seconds slower a lap is under SC / VSC
- `POINTS` — the F1 points system

## Project structure

```
simulate.py       # simulation engine + CLI (the entire program — this is what you run)
requirements.txt  # Python dependencies
schema.sql        # optional PostgreSQL schema for persisting simulation results
.env.example       # example DB connection settings (only relevant if you use schema.sql)
```

`schema.sql` and `.env.example` are for anyone who wants to persist simulation output to a database — `simulate.py` itself never touches a database and needs neither to run.

## Limitations

- `OT_COEFF` (an overtaking-difficulty coefficient per circuit) is defined in `simulate.py` but not currently wired into the simulation — the `--circuit` flag on `run` is purely a display label for now.
- `predict` reuses a generic two-stop `MEDIUM → HARD` strategy for every driver, since the actual strategy for a race that hasn't happened is unknown.
- `DNF_PROB` and `predict`'s baseline circuit-pace fallback are static historical estimates; they don't account for current-season reliability trends or off-season regulation changes.
- OpenF1 rate-limits requests, so `predict --history 40` (the default) can take a few minutes to run.
