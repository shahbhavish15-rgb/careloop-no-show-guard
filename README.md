# CareLoop Clinics — No-Show Guard

**End-to-end Python, SQL, machine-learning and GenAI capstone**

## Business problem
CareLoop Clinics operates eight outpatient clinics across six Indian cities. This project cleans historical appointment records, quantifies no-show patterns, builds a risk model to prioritise next-day follow-up calls, and converts a small sample of call-centre notes into structured reasons and recommended actions.

## Findings from the executed source data
- **Part 1:** 1,020 source booking rows were ingested; 32 failed numeric parsing, 20 exact duplicates were removed, and the final clean table contains 990 rows and 16 columns. Five impossible ages were converted to missing and median-imputed. The final table contains 218 no-shows (22.0%).
- **Part 2:** SQL acceptance checks should return 8 clinics, 990 appointments and 218 no-shows. The 31+ day booking band has a 60.5% no-show rate (26 of 43); the 15–30 day band has a 30.0% rate (76 of 253). The ≥95 appointment clinic table is encoded in `02_queries.sql`.
- **Part 3:** The required 80/20 split produced 792 training rows and 198 test rows, with 49 test no-shows. Logistic Regression had test ROC-AUC 0.789. Training-only cross-validation selected Logistic Regression (mean ROC-AUC 0.774) over the depth-3 tree (0.678), standard Random Forest (0.747), and tuned Random Forest (0.767). The selected out-of-fold threshold was 0.35; on the held-out test set it achieved precision 0.571, recall 0.571 and ROC-AUC 0.789, flagging 49 of 198 appointments.
- **Part 4:** Run `04_genai.ipynb` using your own Gemini API key to generate `outputs/notes_analysed.csv`, validate the ten-note sample and complete the actual recommendation/crosstab. Do not invent AI classifications or upload the key.

## Repository layout

```text
careloop-no-show-guard/
├── data/
│   ├── appointments.csv
│   ├── clinics.csv
│   └── followup_notes.csv
├── outputs/
│   ├── appointments_clean.csv
│   ├── clinics_enriched.csv
│   ├── rejected_rows.json
│   ├── load_data.sql
│   ├── test_risk_scores.csv
│   ├── part1_summary.json
│   └── part3_summary.json
├── 01_ingest_clean.ipynb
├── 02_queries.sql
├── 03_model.ipynb
├── 04_genai.ipynb
└── README.md
```

## Tools
Python, CSV, JSON, pandas, NumPy, Requests, scikit-learn, MySQL 8, Google Gemini API, Pydantic and GitHub.

## How to run
1. Open `01_ingest_clean.ipynb` in Google Colab and upload the three original CSV files when prompted. Run all cells. The notebook creates the cleaned appointment data, rejected-row JSON, clinic enrichment and SQL load script under `outputs/`.
2. In MySQL Workbench, run `02_queries.sql` to create the database/tables, then run `outputs/load_data.sql` and execute the five analysis queries. Record the exact duplicate-key error in the designated SQL comment.
3. Open `03_model.ipynb`, upload `outputs/appointments_clean.csv` if needed, and run all cells. It writes `outputs/test_risk_scores.csv` and model summaries.
4. Open `04_genai.ipynb`. Enter the Gemini API key at runtime only. Run the note-analysis cells; they produce `outputs/notes_analysed.csv`, validation tests and the recommendation analysis.

## Important data and modelling notes
- Original source CSVs are kept unchanged in `data/`.
- The model predicts no-show probability using only information available at the specified evening-before prediction point.
- The test set is used for final reporting only; model and threshold selection are performed on training data using cross-validation.
- The reminder coefficient is an observational association, not proof of causation.
- Open-Meteo weather enrichment must be executed from a network-enabled Colab runtime. If the API fails twice for a clinic, its temperature remains blank/SQL NULL; never invent a temperature.
- The 10 follow-up notes are an illustrative sample, not a clinic-wide estimate of patient reasons.
- Cleaning medians and outlier fences were computed on all 990 cleaned rows before the split; a production pipeline would fit these only on training data.
- The split is random rather than past-to-future, patients may appear across train and test, and a single split limits robustness. The practical threshold also depends on the front desk's available evening call capacity.

## Security
Never commit your Gemini API key. Use runtime input in Colab, clear the input cell output, search the notebook for the key before upload, and verify that no credentials appear in GitHub.

## Submission
Submit only the public GitHub repository URL in the assignment answer field. No Colab link, separate files, screenshots, slides or videos are requested.
