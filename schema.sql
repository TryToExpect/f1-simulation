-- F1 Monte Carlo Simulation — database schema
-- Run: psql -U postgres -d f1_data -f schema.sql

CREATE TABLE IF NOT EXISTS f1_meetings (
    meeting_key            INTEGER PRIMARY KEY,
    meeting_name           TEXT,
    meeting_official_name  TEXT,
    location               TEXT,
    country_name           TEXT,
    circuit_short_name     TEXT,
    year                   INTEGER,
    date_start             TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS f1_sessions (
    session_key   INTEGER PRIMARY KEY,
    meeting_key   INTEGER REFERENCES f1_meetings(meeting_key),
    session_name  TEXT,
    session_type  TEXT,   -- 'Race', 'Qualifying', 'Practice 1/2/3'
    date_start    TIMESTAMPTZ,
    date_end      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS f1_drivers (
    id             SERIAL PRIMARY KEY,
    session_key    INTEGER REFERENCES f1_sessions(session_key),
    driver_number  INTEGER,
    broadcast_name TEXT,
    full_name      TEXT,
    name_acronym   TEXT,
    team_name      TEXT,
    team_colour    TEXT,
    country_code   TEXT,
    UNIQUE (session_key, driver_number)
);

CREATE TABLE IF NOT EXISTS f1_laps (
    id                SERIAL PRIMARY KEY,
    session_key       INTEGER REFERENCES f1_sessions(session_key),
    driver_number     INTEGER,
    lap_number        INTEGER,
    lap_duration      FLOAT,
    duration_sector_1 FLOAT,
    duration_sector_2 FLOAT,
    duration_sector_3 FLOAT,
    i1_speed          INTEGER,
    i2_speed          INTEGER,
    st_speed          INTEGER,
    is_pit_out_lap    BOOLEAN,
    date_start        TIMESTAMPTZ,
    UNIQUE (session_key, driver_number, lap_number)
);

CREATE TABLE IF NOT EXISTS f1_pit_stops (
    id             SERIAL PRIMARY KEY,
    session_key    INTEGER REFERENCES f1_sessions(session_key),
    meeting_key    INTEGER,
    driver_number  INTEGER,
    lap_number     INTEGER,
    pit_duration   FLOAT,
    date           TIMESTAMPTZ,
    UNIQUE (session_key, driver_number, lap_number)
);

CREATE TABLE IF NOT EXISTS f1_stints (
    id                  SERIAL PRIMARY KEY,
    session_key         INTEGER REFERENCES f1_sessions(session_key),
    meeting_key         INTEGER,
    driver_number       INTEGER,
    stint_number        INTEGER,
    compound            TEXT,   -- SOFT / MEDIUM / HARD / INTERMEDIATE / WET
    lap_start           INTEGER,
    lap_end             INTEGER,
    tyre_age_at_start   INTEGER,
    UNIQUE (session_key, driver_number, stint_number)
);

CREATE TABLE IF NOT EXISTS f1_weather (
    id               SERIAL PRIMARY KEY,
    session_key      INTEGER REFERENCES f1_sessions(session_key),
    meeting_key      INTEGER,
    date             TIMESTAMPTZ,
    air_temp         FLOAT,
    track_temp       FLOAT,
    humidity         FLOAT,
    pressure         FLOAT,
    wind_direction   INTEGER,
    wind_speed       FLOAT,
    rainfall         BOOLEAN
);

CREATE TABLE IF NOT EXISTS f1_simulation_results (
    id                    SERIAL PRIMARY KEY,
    run_timestamp         TIMESTAMPTZ DEFAULT NOW(),
    session_key           INTEGER,
    circuit_short_name    TEXT,
    n_simulations         INTEGER,
    driver_number         INTEGER,
    name_acronym          TEXT,
    team_name             TEXT,
    win_pct               FLOAT,
    podium_pct            FLOAT,
    top5_pct              FLOAT,
    top10_pct             FLOAT,
    dnf_pct               FLOAT,
    avg_finish_position   FLOAT,
    points_avg            FLOAT
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_laps_session_driver  ON f1_laps (session_key, driver_number);
CREATE INDEX IF NOT EXISTS idx_pit_session_driver   ON f1_pit_stops (session_key, driver_number);
CREATE INDEX IF NOT EXISTS idx_stints_session       ON f1_stints (session_key, driver_number);
CREATE INDEX IF NOT EXISTS idx_sessions_meeting     ON f1_sessions (meeting_key);
CREATE INDEX IF NOT EXISTS idx_sim_session          ON f1_simulation_results (session_key);
