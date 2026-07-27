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

How to Use ? 

onte Carlo simulator for Formula 1 races. It pulls real session data (lap times, pit stops, tyre strategies) straight from the [OpenF1 API](https://openf1.org) and runs thousands of randomized race simulations to estimate each driver's win probability, podium chances, expected finishing position, and points.

No database is required to run it — everything is fetched live from the API.

## How it works

For a given race session, the script:

1. Fetches drivers, lap times, pit stops, and tyre stints from OpenF1.
2. Builds a per-driver statistical model: median race pace, pace variance, pit stop duration, DNF probability, and tyre strategy.
3. Runs `N` independent race simulations, each lap accounting for:
   - Tyre degradation and the performance cliff per compound (soft/medium/hard/intermediate/wet)
   - Random Safety Car / Virtual Safety Car periods
   - Per-lap DNF risk (from historical team reliability)
   - Pit stop time loss
4. Aggregates the results into win %, podium %, top 5 %, top 10 %, DNF %, average finishing position, and average points per race.

## Requirements

- Python 3.10+
- `requests`
- `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Find a session key

List race sessions for a given year (defaults to 2025):

```bash
python simulate.py list 2025
```

This prints each Grand Prix with its `SESSION_KEY`, date, and circuit — you'll need the key for the next step. Only sessions that have already taken place have data available in OpenF1.

### 2. Run a simulation

```bash
python simulate.py run --session 9912
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--session` | *required* | `session_key` from `list` |
| `--laps` | `57` | Number of race laps |
| `--circuit` | `default` | Circuit name, affects the overtaking model (e.g. `Monza`, `Monaco`) |
| `-n` | `10000` | Number of Monte Carlo iterations |

Example with all options:

```bash
python simulate.py run --session 9574 -n 50000 --laps 57 --circuit Monza
```

### Example output

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

## Model parameters

Tunable constants live at the top of `simulate.py`:

- `TYRE` — degradation rate, cliff lap, and cliff severity per compound
- `OT_COEFF` — overtaking-difficulty coefficient per circuit (`0` = hard to pass, e.g. Monaco; `1` = easy, e.g. Monza)
- `DNF_PROB` — per-team reliability, based on 2022–2024 history
- `P_SC_PER_LAP` / `P_VSC_PER_LAP` — Safety Car / Virtual Safety Car chance per lap
- `POINTS` — F1 points system

## Project structure

```
simulate.py       # simulation engine + CLI
requirements.txt  # Python dependencies
schema.sql         # optional PostgreSQL schema for persisting results (not used by simulate.py)
.env.example        # example DB connection settings (only relevant if you use schema.sql)
```

`schema.sql` and `.env.example` are provided for anyone who wants to persist simulation output to a database — `simulate.py` itself works entirely without one.
