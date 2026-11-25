from pathlib import Path
import csv, json, time
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

# Task 1.1: CSV module ingestion and type validation
with open(DATA / 'appointments.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    raw_rows = list(reader)
print('Task 1.1 headers:', headers)
print('Task 1.1 rows:', len(raw_rows))

def parse_row(row):
    parsed = row.copy()
    parsed['age'] = int(row['age'])
    parsed['fee_inr'] = int(row['fee_inr'])
    parsed['prior_no_shows'] = int(row['prior_no_shows'])
    parsed['distance_km'] = float(row['distance_km'])
    return parsed

accepted, rejected = [], []
for row in raw_rows:
    try:
        accepted.append(parse_row(row))
    except (ValueError, TypeError):
        rejected.append(row)
with open(OUT / 'rejected_rows.json', 'w', encoding='utf-8') as f:
    json.dump(rejected, f, indent=2, ensure_ascii=False)
print('Accepted:', len(accepted), 'Rejected:', len(rejected))

# Task 1.2: start again from the ORIGINAL CSV
src = pd.read_csv(DATA / 'appointments.csv')
print('\nRaw shape:', src.shape)
print('Raw info:')
src.info()
print('Nulls:\n', src.isnull().sum())
print('Exact duplicates:', src.duplicated().sum())

df = src.copy()
df = df.drop_duplicates()
print('After dropping duplicates:', len(df))
df['booking_channel'] = df['booking_channel'].astype('string').str.strip().str.title()
print('Booking channel categories:', sorted(df['booking_channel'].dropna().unique().tolist()))
df['appt_date'] = pd.to_datetime(df['appt_date'], errors='coerce')
df['booked_on'] = pd.to_datetime(df['booked_on'])
df = df.dropna(subset=['appt_date', 'status'])
print('After removing missing/unreadable appt_date and status:', len(df))

# Task 1.3: invalid ages, imputation, outlier capping
invalid_age_mask = (df['age'] < 1) | (df['age'] > 100)
invalid_age_count = int(invalid_age_mask.sum())
df.loc[invalid_age_mask, 'age'] = np.nan
median_age = df['age'].median()
median_distance = df['distance_km'].median()
df['age'] = df['age'].fillna(median_age).round().astype(int)
df['distance_km'] = df['distance_km'].fillna(median_distance)
print('\nInvalid ages:', invalid_age_count, 'Median age:', median_age, 'Median distance:', median_distance)

cap_counts = {}
fences = {}
for col in ['distance_km', 'fee_inr']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    old = df[col].copy()
    df[col] = np.clip(df[col], lower, upper)
    cap_counts[col] = int((old != df[col]).sum())
    fences[col] = {'q1': float(q1), 'q3': float(q3), 'iqr': float(iqr), 'lower': float(lower), 'upper': float(upper)}
df['distance_km'] = df['distance_km'].round(1)
df['fee_inr'] = df['fee_inr'].round().astype(int)
print('Fences:', fences)
print('Capped counts:', cap_counts)

# Task 1.4: features and NumPy summary
df['lead_days'] = (df['appt_date'] - df['booked_on']).dt.days
df['no_show'] = (df['status'] == 'No-Show').astype(int)
X_summary = df[['age', 'distance_km', 'fee_inr', 'lead_days']].to_numpy()
means = np.mean(X_summary, axis=0)
lead_mask = df['lead_days'].to_numpy() > 14
x = df['distance_km'].to_numpy(dtype=float)
z = (x - np.mean(x)) / np.std(x)
z_gt2 = int((np.abs(z) > 2).sum())
print('\nFeature means:', means)
print('Lead days >14:', int(lead_mask.sum()), 'share:', float(lead_mask.mean()))
print('|z(distance)| > 2:', z_gt2)
print('Lead days range:', int(df['lead_days'].min()), int(df['lead_days'].max()))
print('No-shows:', int(df['no_show'].sum()), 'rate:', round(100 * df['no_show'].mean(), 1))

# Task 1.5: live Open-Meteo API enrichment. Set RUN_WEATHER_API=1 to run calls.
import os
clinics = pd.read_csv(DATA / 'clinics.csv')
if os.getenv('RUN_WEATHER_API') == '1':
    temps, api_status = [], []
    url = 'https://api.open-meteo.com/v1/forecast'
    for _, clinic in clinics.iterrows():
        value, reason = None, None
        params = {'latitude': clinic['latitude'], 'longitude': clinic['longitude'], 'hourly': 'temperature_2m', 'forecast_days': 1}
        for attempt in range(2):
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code != 200:
                    reason = f'HTTP {response.status_code}'
                    continue
                payload = response.json()
                arr = payload.get('hourly', {}).get('temperature_2m')
                if not arr:
                    reason = 'hourly.temperature_2m missing or empty'
                    continue
                value = float(arr[0]); reason = None; break
            except requests.exceptions.RequestException as exc:
                reason = str(exc)
        temps.append(value); api_status.append(reason or 'OK')
        if reason: print(f"Weather unavailable for {clinic['clinic_name']}: {reason}")
    clinics['temp_c'] = temps
else:
    # Offline build: do not invent weather. Re-run this notebook with RUN_WEATHER_API=1 in Colab.
    clinics['temp_c'] = None
    api_status = ['NOT_RUN_OFFLINE'] * len(clinics)
clinics.to_csv(OUT / 'clinics_enriched.csv', index=False)

# Task 1.6: exact 16 columns; clean export and SQL literals
cols = ['appointment_id','patient_id','clinic_id','booked_on','appt_date','age','gender','distance_km','fee_inr','booking_channel','visit_type','reminder_sent','prior_no_shows','status','lead_days','no_show']
df = df.sort_values('appointment_id').copy()
for c in ['booked_on', 'appt_date']:
    df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
clean = df[cols].copy()
assert clean['appointment_id'].is_unique
assert not clean.isna().any().any()
clean.to_csv(OUT / 'appointments_clean.csv', index=False)

def sql_value(value):
    if pd.isna(value):
        return 'NULL'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(float(value))
    s = str(value).replace("'", "''")
    return "'" + s + "'"

with open(OUT / 'load_data.sql', 'w', encoding='utf-8') as f:
    f.write('-- Generated from the cleaned original datasets. Clinics inserted before appointments.\n')
    for _, row in clinics.iterrows():
        vals = [row['clinic_id'], row['clinic_name'], row['city'], row['tier'], row['latitude'], row['longitude'], row['temp_c']]
        f.write('INSERT INTO clinics VALUES (' + ','.join(sql_value(v) for v in vals) + ');\n')
    for _, row in clean.iterrows():
        f.write('INSERT INTO appointments VALUES (' + ','.join(sql_value(row[c]) for c in cols) + ');\n')

summary = {
 'raw_rows': len(raw_rows), 'accepted': len(accepted), 'rejected': len(rejected),
 'raw_shape': list(src.shape), 'raw_duplicates': int(src.duplicated().sum()), 'after_dedup': int(len(src.drop_duplicates())),
 'clean_rows': int(len(clean)), 'invalid_ages': invalid_age_count, 'median_age': float(median_age), 'median_distance': float(median_distance),
 'fences': fences, 'capped': cap_counts, 'means': means.tolist(), 'lead_gt14_count': int(lead_mask.sum()), 'lead_gt14_share': float(lead_mask.mean()),
 'distance_abs_z_gt2': z_gt2, 'lead_days_min': int(df['lead_days'].min()), 'lead_days_max': int(df['lead_days'].max()),
 'no_shows': int(clean['no_show'].sum()), 'no_show_rate_pct': round(100*clean['no_show'].mean(),1),
 'weather_status': dict(zip(clinics['clinic_id'], api_status)), 'appointments_unique': bool(clean.appointment_id.is_unique),
 'missing_in_final': int(clean.isna().sum().sum()), 'load_insert_statements': 8 + len(clean)
}
with open(OUT / 'part1_summary.json', 'w', encoding='utf-8') as f: json.dump(summary, f, indent=2)
print('\nPART 1 FINAL SUMMARY')
print(json.dumps(summary, indent=2))
print('Generated:', [p.name for p in OUT.iterdir()])
