-- CARELOOP CLINICS — NO-SHOW GUARD
-- Part 2 SQL deliverable. Run load_data.sql after creating the tables.

-- MySQL 8 setup (skip these two statements when using SQLiteOnline)
CREATE DATABASE IF NOT EXISTS noshowguard;
USE noshowguard;

CREATE TABLE IF NOT EXISTS clinics (
  clinic_id VARCHAR(5) PRIMARY KEY,
  clinic_name VARCHAR(40) NOT NULL,
  city VARCHAR(20) NOT NULL,
  tier VARCHAR(10),
  latitude DECIMAL(8,4),
  longitude DECIMAL(8,4),
  temp_c DECIMAL(5,1)
);

CREATE TABLE IF NOT EXISTS appointments (
  appointment_id VARCHAR(8) PRIMARY KEY,
  patient_id VARCHAR(8) NOT NULL,
  clinic_id VARCHAR(5) NOT NULL,
  booked_on DATE,
  appt_date DATE,
  age INT,
  gender VARCHAR(1),
  distance_km DECIMAL(5,1),
  fee_inr INT,
  booking_channel VARCHAR(10),
  visit_type VARCHAR(12),
  reminder_sent VARCHAR(3),
  prior_no_shows INT,
  status VARCHAR(10),
  lead_days INT,
  no_show INT,
  FOREIGN KEY (clinic_id) REFERENCES clinics(clinic_id)
);

-- Run the generated outputs/load_data.sql file after the table definitions above.
-- Check total records and no-show count:
SELECT COUNT(*) AS clinic_count FROM clinics;
SELECT COUNT(*) AS appointment_count, SUM(no_show) AS total_no_shows FROM appointments;

-- TASK 2.2 — No-show rate by clinic (minimum 95 appointments)
SELECT
    c.clinic_name,
    c.city,
    COUNT(*) AS appointments,
    SUM(a.no_show) AS no_shows,
    ROUND(100.0 * SUM(a.no_show) / COUNT(*), 1) AS no_show_pct
FROM appointments AS a
JOIN clinics AS c ON a.clinic_id = c.clinic_id
GROUP BY c.clinic_id, c.clinic_name, c.city
HAVING COUNT(*) >= 95
ORDER BY no_show_pct DESC;

-- Remove the HAVING line above and rerun to inspect all eight clinics.

-- TASK 2.3 — Does booking far ahead matter?
WITH banded AS (
    SELECT
        CASE
            WHEN lead_days <= 7 THEN '1-7'
            WHEN lead_days <= 14 THEN '8-14'
            WHEN lead_days <= 30 THEN '15-30'
            ELSE '31+'
        END AS lead_band,
        lead_days,
        no_show
    FROM appointments
)
SELECT
    lead_band,
    COUNT(*) AS appointments,
    SUM(no_show) AS no_shows,
    ROUND(100.0 * SUM(no_show) / COUNT(*), 1) AS no_show_pct
FROM banded
GROUP BY lead_band
ORDER BY MIN(lead_days);

-- TASK 2.4 — Month-by-month trend and percentage-point movement
WITH monthly AS (
    SELECT
        SUBSTR(CAST(appt_date AS CHAR), 1, 7) AS month,
        COUNT(*) AS appointments,
        ROUND(100.0 * SUM(no_show) / COUNT(*), 1) AS no_show_pct
    FROM appointments
    GROUP BY SUBSTR(CAST(appt_date AS CHAR), 1, 7)
),
monthly_lag AS (
    SELECT
        month,
        appointments,
        no_show_pct,
        LAG(no_show_pct) OVER (ORDER BY month) AS previous_month_pct
    FROM monthly
)
SELECT
    month,
    appointments,
    no_show_pct,
    ROUND(no_show_pct - previous_month_pct, 1) AS change_pp
FROM monthly_lag
ORDER BY month;

-- TASK 2.5 — Rank clinics within each city and compare with overall rate
WITH clinic_rates AS (
    SELECT
        c.clinic_id,
        c.clinic_name AS clinic,
        c.city,
        ROUND(100.0 * SUM(a.no_show) / COUNT(*), 1) AS no_show_pct
    FROM clinics AS c
    JOIN appointments AS a ON a.clinic_id = c.clinic_id
    GROUP BY c.clinic_id, c.clinic_name, c.city
)
SELECT
    clinic,
    city,
    no_show_pct,
    RANK() OVER (PARTITION BY city ORDER BY no_show_pct DESC) AS rank_in_city,
    CASE
        WHEN no_show_pct > (SELECT 100.0 * SUM(no_show) / COUNT(*) FROM appointments) THEN 'above'
        WHEN no_show_pct < (SELECT 100.0 * SUM(no_show) / COUNT(*) FROM appointments) THEN 'below'
        ELSE 'equal'
    END AS vs_overall
FROM clinic_rates
ORDER BY city, rank_in_city;

-- TASK 2.1 — Duplicate primary-key test (run this statement separately after all queries)
-- Uncomment the INSERT below, execute it by itself, and record the exact client error on the next line.
-- Expected: MySQL duplicate-entry / primary-key error; SQLite UNIQUE constraint failed.
-- INSERT INTO appointments VALUES ('A0001','P509','C05','2025-01-24','2025-01-25',60,'F',18.0,850,'Phone','Consultation','Yes',1,'No-Show',1,1);
-- Duplicate-key error observed: _________________________________________________
