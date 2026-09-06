"""
Dubai Real Estate Forecasting — Streamlit App
==============================================
Forecasts daily transaction volume and average price/sqft for the Dubai
real estate market (DLD/RERA-style data). Ships with synthetic sample data;
swap in real DLD Open Data / Dubai Pulse exports via the sidebar uploader.

Run locally:
    streamlit run app.py

Deploy:
    1. Push this folder to a GitHub repo (app.py, requirements.txt, data/).
    2. Go to https://share.streamlit.io -> "New app".
    3. Point it at the repo, branch, and set "Main file path" to app.py.
"""

import io
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Page config & constants
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dubai Real Estate Forecasting", page_icon="🏙️", layout="wide")

DATA_DIR = "data"
DAILY_SUMMARY_PATH = f"{DATA_DIR}/dld_daily_summary.csv"
TRANSACTIONS_PATH = f"{DATA_DIR}/dld_synthetic_transactions.csv"
RENTALS_PATH = f"{DATA_DIR}/dld_synthetic_rentals.csv"

REQUIRED_COLS = [
    "date", "txn_count", "txn_total_value_aed",
    "txn_avg_price_per_sqft", "rental_count", "rental_avg_annual_rent_aed",
]
LAGS = (7, 14, 30)
ROLLING_WINDOWS = (7, 14, 30)
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

plt.rcParams["axes.grid"] = True
np.random.seed(42)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_base_data():
    daily = pd.read_csv(DAILY_SUMMARY_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    txn = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["transaction_date"])
    try:
        rent = pd.read_csv(RENTALS_PATH, parse_dates=["contract_date"])
    except FileNotFoundError:
        rent = None
    return daily, txn, rent


base_daily_df, txn_df, rent_df = load_base_data()

if "daily_df" not in st.session_state:
    st.session_state.daily_df = base_daily_df.copy()


# ---------------------------------------------------------------------------
# Feature engineering & modeling helpers (cached — re-run automatically
# whenever the underlying data or parameters change)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_features(df, target_col, lags=LAGS, rolling_windows=ROLLING_WINDOWS):
    out = df[["date", target_col]].copy()
    out["dayofweek"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["is_weekend"] = out["dayofweek"].isin([4, 5]).astype(int)  # Fri/Sat = UAE weekend
    out["day_of_year"] = out["date"].dt.dayofyear

    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    for w in rolling_windows:
        out[f"rollmean_{w}"] = out[target_col].shift(1).rolling(w).mean()
        out[f"rollstd_{w}"] = out[target_col].shift(1).rolling(w).std()

    return out.dropna().reset_index(drop=True)


def get_feature_target(feat_df, target_col):
    feature_columns = [c for c in feat_df.columns if c not in ("date", target_col)]
    return feat_df[feature_columns], feat_df[target_col], feature_columns


def time_split(feat_df, test_days):
    split_date = feat_df["date"].max() - pd.Timedelta(days=test_days)
    train = feat_df[feat_df["date"] <= split_date].reset_index(drop=True)
    test = feat_df[feat_df["date"] > split_date].reset_index(drop=True)
    return train, test


def evaluate(y_true, y_pred, label=""):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.nanmean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    return {"label": label, "MAE": mae, "RMSE": rmse, "MAPE": mape}


def pick_best(metrics_a, metrics_b):
    return metrics_a if metrics_a["MAPE"] < metrics_b["MAPE"] else metrics_b


@st.cache_data(show_spinner=False)
def run_sarima(df, target_col, test_days, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    ts = df.set_index("date")[target_col].asfreq("D").interpolate()
    split_date = ts.index.max() - pd.Timedelta(days=test_days)
    train, test = ts[:split_date], ts[split_date + pd.Timedelta(days=1):]

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.get_forecast(steps=len(test))

    return {
        "train": train, "test": test,
        "pred": fc.predicted_mean, "conf_int": fc.conf_int(alpha=0.2),
    }


@st.cache_data(show_spinner=False)
def run_xgboost(df, target_col, test_days):
    feat_df = build_features(df, target_col)
    train, test = time_split(feat_df, test_days)
    X_train, y_train, feature_columns = get_feature_target(train, target_col)
    X_test, y_test, _ = get_feature_target(test, target_col)

    model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model": model, "feature_columns": feature_columns,
        "y_test": y_test, "preds": preds, "test_dates": test["date"],
    }


def forecast_future_xgb(df, target_col, model_bundle, horizon, n_boot=200):
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    residuals = model_bundle["y_test"].values - model_bundle["preds"]

    history = df[["date", target_col]].copy()
    future_preds, future_dates = [], []
    last_date = history["date"].max()

    for step in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=step)
        row = {
            "dayofweek": next_date.dayofweek,
            "month": next_date.month,
            "is_weekend": int(next_date.dayofweek in [4, 5]),
            "day_of_year": next_date.dayofyear,
        }
        for lag in LAGS:
            row[f"lag_{lag}"] = history[target_col].iloc[-lag] if len(history) >= lag else history[target_col].iloc[-1]
        for w in ROLLING_WINDOWS:
            window = history[target_col].iloc[-w:]
            row[f"rollmean_{w}"] = window.mean()
            row[f"rollstd_{w}"] = window.std() if len(window) > 1 else 0.0

        X_next = pd.DataFrame([row])[feature_columns]
        pred = float(model.predict(X_next)[0])

        future_preds.append(pred)
        future_dates.append(next_date)
        history = pd.concat([history, pd.DataFrame({"date": [next_date], target_col: [pred]})], ignore_index=True)

    future_preds = np.array(future_preds)
    lo, hi = [], []
    for p in future_preds:
        samples = p + np.random.choice(residuals, size=n_boot, replace=True)
        lo.append(np.percentile(samples, 10))
        hi.append(np.percentile(samples, 90))

    return pd.DataFrame({"date": future_dates, "forecast": future_preds, "lower_80": lo, "upper_80": hi})


def forecast_future_sarima(df, target_col, horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    ts = df.set_index("date")[target_col].asfreq("D").interpolate()
    fit = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = fit.get_forecast(steps=horizon)
    mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.2)
    return pd.DataFrame({
        "date": mean.index, "forecast": mean.values,
        "lower_80": ci.iloc[:, 0].values, "upper_80": ci.iloc[:, 1].values,
    })


def forecast_forward(df, target_col, best_metrics, xgb_bundle, horizon):
    if "XGBoost" in best_metrics["label"]:
        return forecast_future_xgb(df, target_col, xgb_bundle, horizon)
    return forecast_future_sarima(df, target_col, horizon)


# ---------------------------------------------------------------------------
# Sidebar — settings + add new data
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
test_days = st.sidebar.slider("Holdout test period (days)", 30, 120, 60, step=10)
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 60, 30, step=1)
top_n_areas = st.sidebar.slider("Areas to forecast individually", 1, 8, 3)

st.sidebar.markdown("---")
st.sidebar.header("📥 Add New Data")
st.sidebar.caption(
    "Upload a CSV with the same columns as the daily summary: "
    "`date, txn_count, txn_total_value_aed, txn_avg_price_per_sqft, "
    "rental_count, rental_avg_annual_rent_aed`. Rows with a matching "
    "`date` overwrite existing ones."
)
uploaded_file = st.sidebar.file_uploader("Choose CSV", type="csv")
if uploaded_file is not None and st.sidebar.button("➕ Add Data"):
    try:
        new_data = pd.read_csv(uploaded_file, parse_dates=["date"])
        missing = set(REQUIRED_COLS) - set(new_data.columns)
        if missing:
            st.sidebar.error(f"Missing columns: {missing}")
        else:
            combined = pd.concat(
                [st.session_state.daily_df, new_data[REQUIRED_COLS]], ignore_index=True
            )
            combined = (combined.drop_duplicates(subset="date", keep="last")
                                 .sort_values("date").reset_index(drop=True))
            st.session_state.daily_df = combined
            st.sidebar.success(
                f"Added {len(new_data)} row(s). Dataset now spans "
                f"{combined['date'].min().date()} → {combined['date'].max().date()}."
            )
    except Exception as e:
        st.sidebar.error(f"Could not process file: {e}")

if st.sidebar.button("↩️ Reset to sample data"):
    st.session_state.daily_df = base_daily_df.copy()
    st.sidebar.info("Reset to the bundled sample dataset.")

daily_df = st.session_state.daily_df

st.sidebar.download_button(
    "⬇️ Download current dataset",
    data=daily_df.to_csv(index=False).encode("utf-8"),
    file_name="daily_summary_current.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Main body
# ---------------------------------------------------------------------------
st.title("🏙️ Dubai Real Estate Forecasting")
st.caption("Transaction volume & price/sqft forecasting — SARIMA vs. XGBoost, on DLD/RERA-style data")
st.write(
    f"**Current dataset:** {len(daily_df)} days · "
    f"{daily_df['date'].min().date()} → {daily_df['date'].max().date()}"
)

tab_overview, tab_eda, tab_models, tab_areas, tab_forecast = st.tabs(
    ["📊 Overview", "🔍 EDA", "🤖 Models", "📍 Area Forecasts", "🔮 Forward Forecast"]
)

# --- Overview -----------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg daily transactions", f"{daily_df['txn_count'].mean():.1f}")
    c2.metric("Avg price/sqft (AED)", f"{daily_df['txn_avg_price_per_sqft'].mean():,.0f}")
    c3.metric("Avg daily rentals", f"{daily_df['rental_count'].mean():.1f}")
    c4.metric("Avg annual rent (AED)", f"{daily_df['rental_avg_annual_rent_aed'].mean():,.0f}")

    st.subheader("Recent data")
    recent = daily_df.tail(20).copy()
    recent["date"] = recent["date"].dt.date
    st.dataframe(recent, use_container_width=True)

    with st.expander("Summary statistics"):
        st.dataframe(daily_df.drop(columns="date").describe(), use_container_width=True)

    missing_cols = daily_df.isna().sum()
    full_range = pd.date_range(daily_df["date"].min(), daily_df["date"].max(), freq="D")
    missing_days = full_range.difference(daily_df["date"])
    with st.expander("Data quality checks"):
        st.write("Missing values per column:")
        st.dataframe(missing_cols.rename("missing_count"))
        st.write(f"Missing calendar days: **{len(missing_days)}**")

# --- EDA ------------------------------------------------------------------
with tab_eda:
    st.subheader("Trend & Weekly Seasonality Decomposition")
    ts = daily_df.set_index("date")["txn_count"].asfreq("D").interpolate()
    decomposition = seasonal_decompose(ts, model="additive", period=7)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    decomposition.observed.plot(ax=axes[0], title="Observed — Daily Transaction Count")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Weekly Seasonality")
    decomposition.resid.plot(ax=axes[3], title="Residual (noise)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        monthly = daily_df.assign(month=daily_df["date"].dt.month).groupby("month")["txn_count"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        monthly.plot(kind="bar", ax=ax, title="Avg Daily Transactions by Month")
        ax.set_ylabel("Avg transactions/day")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        dow = (daily_df.assign(dow=daily_df["date"].dt.day_name())
               .groupby("dow")["txn_count"].mean().reindex(DOW_ORDER))
        fig, ax = plt.subplots(figsize=(6, 4))
        dow.plot(kind="bar", ax=ax, title="Avg Daily Transactions by Day of Week", color="tab:orange")
        ax.set_ylabel("Avg transactions/day")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Price Trend")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_df["date"], daily_df["txn_avg_price_per_sqft"], label="Avg price/sqft (AED)")
    ax.plot(daily_df["date"], daily_df["txn_avg_price_per_sqft"].rolling(30).mean(),
            label="30-day rolling avg", color="tab:red")
    ax.set_title("Average Transaction Price per Sqft Over Time")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# --- Models -----------------------------------------------------------
with tab_models:
    st.subheader("Train & Compare Models")
    st.caption(
        "SARIMA (classical, captures weekly seasonality natively) vs. XGBoost "
        "(ML on calendar + lag + rolling features), evaluated on a chronological holdout."
    )

    if st.button("🚀 Train models", type="primary"):
        with st.spinner("Training SARIMA & XGBoost..."):
            sarima_txn = run_sarima(daily_df, "txn_count", test_days)
            sarima_price = run_sarima(daily_df, "txn_avg_price_per_sqft", test_days)
            xgb_txn = run_xgboost(daily_df, "txn_count", test_days)
            xgb_price = run_xgboost(daily_df, "txn_avg_price_per_sqft", test_days)

            m1 = evaluate(sarima_txn["test"].values, sarima_txn["pred"].values, "SARIMA - txn_count")
            m2 = evaluate(xgb_txn["y_test"].values, xgb_txn["preds"], "XGBoost - txn_count")
            m3 = evaluate(sarima_price["test"].values, sarima_price["pred"].values, "SARIMA - avg_price_sqft")
            m4 = evaluate(xgb_price["y_test"].values, xgb_price["preds"], "XGBoost - avg_price_sqft")

            st.session_state["model_results"] = {
                "sarima_txn": sarima_txn, "sarima_price": sarima_price,
                "xgb_txn": xgb_txn, "xgb_price": xgb_price,
                "metrics": [m1, m2, m3, m4],
                "best_txn": pick_best(m1, m2), "best_price": pick_best(m3, m4),
                "test_days": test_days,
            }

    if "model_results" in st.session_state:
        res = st.session_state["model_results"]
        results_df = pd.DataFrame(res["metrics"])
        st.dataframe(results_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}%"}),
                     use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        results_df.set_index("label")[["MAPE"]].plot(kind="bar", legend=False, ax=ax, title="Model Comparison — MAPE (%)")
        ax.set_ylabel("MAPE %")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.success(
            f"**Best for transaction count:** {res['best_txn']['label']} "
            f"(MAPE={res['best_txn']['MAPE']:.2f}%)  \n"
            f"**Best for avg price/sqft:** {res['best_price']['label']} "
            f"(MAPE={res['best_price']['MAPE']:.2f}%)"
        )

        col1, col2 = st.columns(2)
        with col1:
            sarima_txn = res["sarima_txn"]
            fig, ax = plt.subplots(figsize=(6, 4))
            sarima_txn["train"][-90:].plot(ax=ax, label="Train (last 90d)")
            sarima_txn["test"].plot(ax=ax, label="Actual (test)")
            sarima_txn["pred"].plot(ax=ax, label="SARIMA forecast", style="--")
            ax.fill_between(sarima_txn["conf_int"].index, sarima_txn["conf_int"].iloc[:, 0],
                             sarima_txn["conf_int"].iloc[:, 1], alpha=0.2)
            ax.set_title("SARIMA — Transaction Count (Test)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            xgb_txn = res["xgb_txn"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xgb_txn["test_dates"], xgb_txn["y_test"].values, label="Actual")
            ax.plot(xgb_txn["test_dates"], xgb_txn["preds"], "--", label="XGBoost forecast")
            ax.set_title("XGBoost — Transaction Count (Test)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("XGBoost feature importance (txn_count)"):
            importances = pd.Series(res["xgb_txn"]["model"].feature_importances_,
                                     index=res["xgb_txn"]["feature_columns"]).sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            importances.plot(kind="barh", ax=ax, title="XGBoost Feature Importance — txn_count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("Residual analysis (XGBoost — txn_count)"):
            resid = res["xgb_txn"]["y_test"].values - res["xgb_txn"]["preds"]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(res["xgb_txn"]["test_dates"], resid)
            axes[0].axhline(0, color="red", ls="--")
            axes[0].set_title("Residuals Over Time")
            axes[1].hist(resid, bins=20)
            axes[1].set_title("Residual Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Click **Train models** to run SARIMA and XGBoost on the current dataset.")

# --- Area forecasts -----------------------------------------------------
with tab_areas:
    st.subheader(f"Top {top_n_areas} Areas by Transaction Volume")
    st.caption("Same XGBoost pipeline applied per area — shows the approach works below citywide granularity.")

    if st.button("🚀 Run area-level forecasts", type="primary"):
        with st.spinner("Training per-area models..."):
            area_daily = (txn_df.groupby([txn_df["transaction_date"].dt.floor("D"), "area_name"])
                          .size().rename("txn_count").reset_index()
                          .rename(columns={"transaction_date": "date"}))

            top_areas = (area_daily.groupby("area_name")["txn_count"].sum()
                         .sort_values(ascending=False).head(top_n_areas).index.tolist())

            full_dates = pd.date_range(daily_df["date"].min(), daily_df["date"].max(), freq="D")
            area_results = {}
            area_metrics = []

            for area in top_areas:
                a_df = area_daily[area_daily["area_name"] == area][["date", "txn_count"]]
                a_df = (a_df.set_index("date").reindex(full_dates, fill_value=0)
                        .rename_axis("date").reset_index())
                xgb_area = run_xgboost(a_df, "txn_count", test_days)
                m = evaluate(xgb_area["y_test"].values, xgb_area["preds"], f"XGBoost - {area}")
                area_results[area] = xgb_area
                area_metrics.append(m)

            st.session_state["area_results"] = {"results": area_results, "metrics": area_metrics, "areas": top_areas}

    if "area_results" in st.session_state:
        ar = st.session_state["area_results"]
        st.dataframe(pd.DataFrame(ar["metrics"]).style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}%"}),
                     use_container_width=True)

        for area in ar["areas"]:
            r = ar["results"][area]
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(r["test_dates"], r["y_test"].values, label="Actual")
            ax.plot(r["test_dates"], r["preds"], "--", label="Forecast")
            ax.set_title(area)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        st.caption(
            "Note: MAPE can look inflated for lower-volume areas because a few "
            "transactions' difference is a large percentage of a small daily count."
        )
    else:
        st.info("Click **Run area-level forecasts** to segment by neighborhood.")

# --- Forward forecast -----------------------------------------------------
with tab_forecast:
    st.subheader(f"{horizon}-Day Forward Forecast (80% Confidence Interval)")

    if "model_results" not in st.session_state:
        st.info("Train models in the **🤖 Models** tab first — the forward forecast uses the winning model per target.")
    else:
        if st.button("🔮 Generate forward forecast", type="primary"):
            res = st.session_state["model_results"]
            with st.spinner("Forecasting forward..."):
                future_txn = forecast_forward(daily_df, "txn_count", res["best_txn"], res["xgb_txn"], horizon)
                future_price = forecast_forward(daily_df, "txn_avg_price_per_sqft", res["best_price"], res["xgb_price"], horizon)
                st.session_state["future_forecast"] = {"txn": future_txn, "price": future_price}

        if "future_forecast" in st.session_state:
            res = st.session_state["model_results"]
            ff = st.session_state["future_forecast"]

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(daily_df["date"][-90:], daily_df["txn_count"][-90:], label="History (last 90d)")
            ax.plot(ff["txn"]["date"], ff["txn"]["forecast"], "--", color="tab:orange",
                    label=f"Forecast ({res['best_txn']['label']})")
            ax.fill_between(ff["txn"]["date"], ff["txn"]["lower_80"], ff["txn"]["upper_80"],
                             color="tab:orange", alpha=0.2, label="80% CI")
            ax.set_title(f"{horizon}-Day Forward Forecast — Daily Transaction Count")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(daily_df["date"][-90:], daily_df["txn_avg_price_per_sqft"][-90:], label="History (last 90d)")
            ax.plot(ff["price"]["date"], ff["price"]["forecast"], "--", color="tab:green",
                    label=f"Forecast ({res['best_price']['label']})")
            ax.fill_between(ff["price"]["date"], ff["price"]["lower_80"], ff["price"]["upper_80"],
                             color="tab:green", alpha=0.2, label="80% CI")
            ax.set_title(f"{horizon}-Day Forward Forecast — Avg Price per Sqft (AED)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download transaction forecast",
                    data=ff["txn"].to_csv(index=False).encode("utf-8"),
                    file_name="txn_count_forecast.csv", mime="text/csv",
                )
            with col2:
                st.download_button(
                    "⬇️ Download price forecast",
                    data=ff["price"].to_csv(index=False).encode("utf-8"),
                    file_name="price_forecast.csv", mime="text/csv",
                )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Sample data is synthetic (Dubai Land Department / RERA style). "
    "Swap in real data via the uploader above once ready."
)
