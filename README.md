# Dubai Real Estate Forecasting — Streamlit App

Forecasts daily transaction volume and average price/sqft for the Dubai
real estate market (DLD/RERA-style data), comparing SARIMA vs. XGBoost.
Ships with synthetic sample data so it runs out of the box.

## Features
- 📊 Data overview & quality checks
- 🔍 EDA: trend, weekly seasonality (UAE Fri/Sat weekend), monthly seasonality
- 🤖 SARIMA vs. XGBoost, evaluated with MAE / RMSE / MAPE on a time-based holdout
- 📍 Area-level forecasts for the top N neighborhoods by volume
- 🔮 30-day (configurable) forward forecast with an 80% confidence interval
- 📥 Upload new daily data from the sidebar and re-run everything on it
- ⬇️ Download the current dataset and forecasts as CSV

## Project structure
```
.
├── app.py                 # the Streamlit app
├── requirements.txt
├── README.md
└── data/
    ├── dld_daily_summary.csv          # citywide daily aggregates
    ├── dld_synthetic_transactions.csv # row-level transactions (used for area breakdown)
    └── dld_synthetic_rentals.csv      # row-level rental contracts (reference data)
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Deploy to Streamlit Community Cloud

1. **Push this folder to a new GitHub repo** (keep `app.py`, `requirements.txt`,
   and the `data/` folder at the paths shown above):
   ```bash
   git init
   git add .
   git commit -m "Initial commit — Dubai real estate forecasting app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, pick your repo and branch, and set:
   - **Main file path:** `app.py`
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically
   and gives you a public URL.

## Using your own data
Replace the CSVs in `data/` before deploying (keep the same filenames and
columns), or use the **"Add New Data"** uploader in the running app's
sidebar to append/overwrite days without redeploying — uploaded data lives
only in that browser session's memory, so it resets if the app restarts.

Required columns for `dld_daily_summary.csv` (and any uploaded CSV):
`date, txn_count, txn_total_value_aed, txn_avg_price_per_sqft, rental_count, rental_avg_annual_rent_aed`

`dld_synthetic_transactions.csv` needs at least `transaction_date` and
`area_name` for the area-level forecast tab to work.

## Notes
- All sample data is **synthetic** — no real transactions or individuals.
- Model training is cached (`st.cache_data`) so switching tabs doesn't
  retrain unless the data or settings actually change.
- For real DLD data, see https://dubailand.gov.ae/en/open-data/real-estate-data/
  or Dubai Pulse.
