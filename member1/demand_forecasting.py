"""
TripSense - Member 1: Demand Forecasting + Hotel Intelligence
Pipeline: hotel_bookings.csv -> cleaning -> feature engineering -> Linear Regression (baseline)
          -> Random Forest -> XGBoost -> evaluation (MAE/RMSE) -> SHAP -> hotel pricing

Target: daily occupied room-nights per hotel (City / Resort), forecast 7 days ahead.
Usage:  python demand_forecasting.py [path/to/hotel_bookings.csv]
"""
import sys, os, json, joblib, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import holidays, shap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "hotel_bookings.csv"
OUT = "."
for _d in ("data", "plots", "models"):
    os.makedirs(f"{OUT}/{_d}", exist_ok=True)
HORIZON = 7            # forecast horizon in days (lags are >= HORIZON, so no leakage)
TEST_DAYS = 120        # last 120 days held out (time-based split)
MONTHS = {m: i for i, m in enumerate(["January","February","March","April","May","June","July",
          "August","September","October","November","December"], 1)}

# ------------------------------------------------------------------ 1. DATA CLEANING
raw = pd.read_csv(CSV)
df = raw.copy()
log = {"raw_rows": len(raw)}
df["is_canceled"] = df["is_canceled"].map({"yes": 1, "no": 0, 1: 1, 0: 0}).astype(int)
df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")
for c in ["agent", "company"]:                      # 'NULL' strings -> NaN (agent/company ids)
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["total_guests"] = df.adults + df.children + df.babies
df["total_nights"] = df.stays_in_weekend_nights + df.stays_in_week_nights
bad = (df.total_guests == 0) | (df.total_nights == 0) | (df.adr < 0) | (df.adr > 1000)
log["dropped_invalid_rows"] = int(bad.sum())        # 0 guests, 0 nights, negative / extreme ADR
df = df[~bad].copy()
df["arrival_date"] = pd.to_datetime(dict(year=df.arrival_date_year,
                     month=df.arrival_date_month.map(MONTHS), day=df.arrival_date_day_of_month))
log["clean_rows"] = len(df)
log["cancellation_rate_pct"] = round(df.is_canceled.mean() * 100, 2)

# ------------------------------------------------------------------ 2. DAILY DEMAND TABLE
# Demand = room-nights actually occupied (non-cancelled bookings expanded over every night of stay)
stay = df[df.is_canceled == 0].copy()
stay = stay.loc[stay.index.repeat(stay.total_nights)]
stay["night"] = stay.arrival_date + pd.to_timedelta(stay.groupby(level=0).cumcount(), unit="D")
daily = (stay.groupby(["hotel", "night"])
             .agg(demand=("adr", "size"), adr=("adr", "mean")).reset_index())
arr = df.groupby(["hotel", "arrival_date"]).agg(
        bookings=("is_canceled", "size"), cancels=("is_canceled", "sum")).reset_index()
arr["cancel_rate"] = arr.cancels / arr.bookings
last_arrival = df.arrival_date.max()
frames = []
for h, g in daily.groupby("hotel"):                 # full date range, fill gaps with 0
    idx = pd.date_range(g.night.min() + pd.Timedelta(days=30), last_arrival)  # drop start ramp-up
    g = g.set_index("night").reindex(idx)
    g["demand"] = g.demand.fillna(0); g["adr"] = g.adr.ffill()
    g["hotel"] = h; g.index.name = "date"; frames.append(g.reset_index())
ts = pd.concat(frames, ignore_index=True)
ts = ts.merge(arr[["hotel", "arrival_date", "cancel_rate"]], left_on=["hotel", "date"],
              right_on=["hotel", "arrival_date"], how="left").drop(columns="arrival_date")
# occupancy index: demand relative to the hotel's peak (dataset has no room capacity column)
cap = ts.groupby("hotel").demand.quantile(0.99)
ts["capacity_proxy"] = ts.hotel.map(cap)
ts["occupancy_pct"] = (ts.demand / ts.capacity_proxy * 100).clip(upper=100)

# ------------------------------------------------------------------ 3. FEATURE ENGINEERING
pt_hol = holidays.country_holidays("PT", years=range(2015, 2019))
in_hol = holidays.country_holidays("IN", years=range(2015, 2019))   # India calendar (shared w/ Member 3)
hol_dates = pd.to_datetime(sorted(pt_hol.keys()))

def add_features(d):
    d = d.sort_values(["hotel", "date"]).copy()
    d["month"] = d.date.dt.month; d["dayofweek"] = d.date.dt.dayofweek
    d["dayofyear"] = d.date.dt.dayofyear; d["weekofyear"] = d.date.dt.isocalendar().week.astype(int)
    d["is_weekend"] = (d.dayofweek >= 5).astype(int)
    d["is_summer"] = d.month.isin([6, 7, 8]).astype(int)
    d["sin_doy"] = np.sin(2 * np.pi * d.dayofyear / 365.25); d["cos_doy"] = np.cos(2 * np.pi * d.dayofyear / 365.25)
    d["is_pt_holiday"] = d.date.isin(hol_dates).astype(int)
    d["is_in_holiday"] = d.date.apply(lambda x: int(x in in_hol))
    nxt = np.searchsorted(hol_dates.values, d.date.values)
    d["days_to_holiday"] = [(hol_dates[min(i, len(hol_dates)-1)] - t).days for i, t in zip(nxt, d.date)]
    d["days_to_holiday"] = d.days_to_holiday.clip(0, 30)
    d["is_city"] = (d.hotel == "City Hotel").astype(int)
    g = d.groupby("hotel")
    for L in (HORIZON, 14, 21, 28):                  # lags >= horizon: usable at forecast time
        d[f"lag_{L}"] = g.demand.shift(L)
    sh = g.demand.shift(HORIZON)
    d["roll_mean_7"] = sh.groupby(d.hotel).transform(lambda s: s.rolling(7).mean())
    d["roll_mean_28"] = sh.groupby(d.hotel).transform(lambda s: s.rolling(28).mean())
    d["roll_std_7"] = sh.groupby(d.hotel).transform(lambda s: s.rolling(7).std())
    d["adr_lag_7"] = g.adr.shift(HORIZON)
    return d.dropna(subset=["lag_28", "roll_mean_28"])

feat = add_features(ts)
FEATURES = ["month", "dayofweek", "dayofyear", "weekofyear", "is_weekend", "is_summer", "sin_doy", "cos_doy",
            "is_pt_holiday", "is_in_holiday", "days_to_holiday", "is_city", "lag_7", "lag_14", "lag_21",
            "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_7", "adr_lag_7"]
cut = feat.date.max() - pd.Timedelta(days=TEST_DAYS)
train, test = feat[feat.date <= cut], feat[feat.date > cut]
Xtr, ytr, Xte, yte = train[FEATURES], train.demand, test[FEATURES], test.demand
print(f"Train {len(train)} rows (to {cut.date()}) | Test {len(test)} rows ({test.date.min().date()} -> {test.date.max().date()})")

# ------------------------------------------------------------------ 4. MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=2, random_state=42, n_jobs=-1),
}
naive = test.lag_7.values                            # seasonal-naive reference (same weekday last week)
rows, preds = [], {}
rows.append({"Model": "Naive (lag-7)", "MAE": mean_absolute_error(yte, naive),
             "RMSE": mean_squared_error(yte, naive) ** .5, "R2": r2_score(yte, naive)})
for name, m in models.items():
    m.fit(Xtr, ytr); p = np.clip(m.predict(Xte), 0, None); preds[name] = p
    rows.append({"Model": name, "MAE": mean_absolute_error(yte, p),
                 "RMSE": mean_squared_error(yte, p) ** .5, "R2": r2_score(yte, p)})
    for h in ["City Hotel", "Resort Hotel"]:
        k = (test.hotel == h).values
        rows.append({"Model": f"  {name} [{h}]", "MAE": mean_absolute_error(yte[k], p[k]),
                     "RMSE": mean_squared_error(yte[k], p[k]) ** .5, "R2": r2_score(yte[k], p[k])})
metrics = pd.DataFrame(rows).round(3)
metrics.to_csv(f"{OUT}/data/model_metrics.csv", index=False)
print(metrics.to_string(index=False))
best = min(models, key=lambda n: mean_squared_error(yte, preds[n]))
print("Best model:", best)

# ------------------------------------------------------------------ 5. FORECAST vs ACTUAL
test = test.assign(**{f"pred_{n}": p for n, p in preds.items()})
test["pred_best"] = preds[best]
test["pred_occupancy_pct"] = (test.pred_best / test.capacity_proxy * 100).clip(upper=100)
test[["hotel", "date", "demand", "occupancy_pct", "pred_best", "pred_occupancy_pct",
      *[f"pred_{n}" for n in models]]].to_csv(f"{OUT}/data/forecast_vs_actual.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for ax, h in zip(axes, ["City Hotel", "Resort Hotel"]):
    t = test[test.hotel == h]
    ax.plot(t.date, t.demand, "k-", lw=2, label="Actual")
    ax.plot(t.date, t["pred_Linear Regression"], ":", c="tab:gray", label="Linear Reg.")
    ax.plot(t.date, t["pred_Random Forest"], "--", c="tab:green", label="Random Forest")
    ax.plot(t.date, t["pred_XGBoost"], "-", c="tab:red", lw=1.6, label="XGBoost")
    ax.set_title(f"{h} - 7-day-ahead forecast vs actual (room-nights/day)"); ax.legend(ncol=4); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/plots/forecast_vs_actual.png", dpi=150); plt.close()

mm = metrics[~metrics.Model.str.startswith("  ")].set_index("Model")
ax = mm[["MAE", "RMSE"]].plot.bar(figsize=(8, 4.5), rot=15, color=["#4c78a8", "#e45756"])
ax.set_title("Model comparison (lower is better)"); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/plots/model_comparison.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 6. SHAP EXPLANATION (XGBoost)
xgb = models["XGBoost"]
expl = shap.TreeExplainer(xgb)
sv = expl(Xte)
plt.figure(); shap.summary_plot(sv, Xte, show=False, max_display=15); plt.tight_layout()
plt.savefig(f"{OUT}/plots/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
plt.figure(); shap.plots.bar(sv, show=False, max_display=15); plt.tight_layout()
plt.savefig(f"{OUT}/plots/shap_importance.png", dpi=150, bbox_inches="tight"); plt.close()
imp = pd.Series(np.abs(sv.values).mean(0), index=FEATURES).sort_values(ascending=False)
imp.rename("mean_abs_shap").to_csv(f"{OUT}/data/shap_importance.csv")
print("\nTop SHAP drivers:\n", imp.head(8).round(2))

# ------------------------------------------------------------------ 7. HOTEL PRICING
# Dynamic price = hotel's seasonal base ADR x demand multiplier derived from predicted occupancy.
base = ts.groupby(["hotel", ts.date.dt.month]).adr.median().rename("base_adr").reset_index()
base.columns = ["hotel", "month", "base_adr"]
p = test.merge(base, on=["hotel", "month"])
avg_occ = p.groupby("hotel").pred_occupancy_pct.transform("mean")
p["demand_index"] = p.pred_occupancy_pct / avg_occ
p["price_multiplier"] = (1 + 0.5 * (p.demand_index - 1)).clip(0.80, 1.30)    # +-: capped -20% / +30%
p["recommended_price"] = (p.base_adr * p.price_multiplier).round(2)
p["demand_level"] = pd.cut(p.pred_occupancy_pct, [-1, 40, 70, 101], labels=["Low", "Medium", "High"])
p[["hotel", "date", "pred_best", "pred_occupancy_pct", "demand_level", "base_adr",
   "price_multiplier", "recommended_price"]].round(2).to_csv(f"{OUT}/data/hotel_pricing.csv", index=False)

# ---- future forecast (next 7 days) for the Streamlit dashboard, using recursive prediction
fut_rows = []
hist = ts.copy()
for h in ["City Hotel", "Resort Hotel"]:
    hh = hist[hist.hotel == h].copy()
    for step in range(HORIZON):                       # recursive: predicted values feed later lags
        nd = hh.date.max() + pd.Timedelta(days=1)
        hh = pd.concat([hh, pd.DataFrame({"date": [nd], "hotel": [h], "demand": [np.nan],
                       "adr": [hh.adr.iloc[-1]], "capacity_proxy": [hh.capacity_proxy.iloc[0]]})], ignore_index=True)
        f = add_features(hh.assign(demand=hh.demand.fillna(0))).tail(1)   # lags >= 7 never touch the placeholder
        yhat = max(0, float(models[best].predict(f[FEATURES])[0]))
        hh.loc[hh.index[-1], "demand"] = yhat
        fut_rows.append({"hotel": h, "date": nd, "pred_demand": round(yhat, 1),
                         "pred_occupancy_pct": round(min(100, yhat / hh.capacity_proxy.iloc[0] * 100), 1)})
pd.DataFrame(fut_rows).to_csv(f"{OUT}/data/future_forecast_7d.csv", index=False)

# ------------------------------------------------------------------ 8. EXTRA HOTEL INTELLIGENCE
seas = (ts.assign(month=ts.date.dt.month).groupby(["hotel", "month"])
          .agg(avg_demand=("demand", "mean"), avg_occupancy_pct=("occupancy_pct", "mean"), avg_adr=("adr", "mean"))
          .round(2).reset_index())
seas.to_csv(f"{OUT}/data/monthly_seasonality.csv", index=False)
dow = (ts.assign(dow=ts.date.dt.day_name()).groupby(["hotel", "dow"]).demand.mean().round(1).reset_index())
dow.to_csv(f"{OUT}/data/weekday_pattern.csv", index=False)
ts.to_csv(f"{OUT}/data/daily_demand_clean.csv", index=False)

# ------------------------------------------------------------------ 9. SAVE ARTIFACTS
joblib.dump({"model": models[best], "xgboost": xgb, "features": FEATURES, "best": best}, f"{OUT}/models/xgboost_demand.pkl")
joblib.dump(models["Random Forest"], f"{OUT}/models/random_forest_demand.pkl")
joblib.dump(models["Linear Regression"], f"{OUT}/models/linear_regression_demand.pkl")
log.update({"daily_rows": len(ts), "best_model": best, "test_window": [str(test.date.min().date()), str(test.date.max().date())],
            "metrics": mm.round(2).to_dict("index")})
json.dump(log, open(f"{OUT}/data/run_summary.json", "w"), indent=2)
print("\nDone. Artifacts saved in", OUT)
