# Final submission checklist

- [ ] `data/appointments.csv`, `data/clinics.csv`, `data/followup_notes.csv` are the original files and remain unchanged.
- [ ] `01_ingest_clean.ipynb` runs end-to-end in Colab; weather calls return real values or explicitly logged failures (never invented values).
- [ ] `outputs/rejected_rows.json` contains 32 rejected rows.
- [ ] `outputs/appointments_clean.csv` contains 990 rows, 16 columns, unique appointment IDs and no missing values.
- [ ] `outputs/clinics_enriched.csv` contains 8 clinics plus `temp_c`.
- [ ] `outputs/load_data.sql` contains 998 INSERT statements, clinics before appointments.
- [ ] `02_queries.sql` contains all five SQL business questions and the recorded duplicate-key error.
- [ ] SQL checks return 8 clinics, 990 appointments and 218 no-shows.
- [ ] `03_model.ipynb` runs; split is 792/198, model selection and threshold selection use training-only CV.
- [ ] `outputs/test_risk_scores.csv` contains 198 rows and the chosen threshold/flag result.
- [ ] `04_genai.ipynb` runs with a runtime-only Gemini key; N004 is skipped without an API call; invalid replies are retained.
- [ ] `outputs/notes_analysed.csv` contains 10 rows and the required columns.
- [ ] README has four numerical findings, a business recommendation, model/AUC/threshold and three limitations.
- [ ] GitHub repository is PUBLIC and all required files are uploaded.
- [ ] Gemini key is absent from every committed file and notebook output.
- [ ] Paste ONLY the public GitHub repository URL into the LMS answer box.
