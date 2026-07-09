# predict_v2.py - GB peak-demand forecast WITH lag features, via live Elexon fetch.
# Graceful: if the demand fetch fails or data isn't published yet, it logs a
# "skipped" row and exits 0 (so the Action still passes). Picks up again next run.
import os
import datetime as dt
import pandas as pd
import requests
import lightgbm as lgb
import holidays

LAT, LON = 51.51, -0.13
LOG = "predictions_log.csv"
FEATURES = ["t_mean","t_max","t_min","HDD","CDD",
            "dow","is_we","month","doy","is_hol","lag1","lag7"]

today = pd.Timestamp.now(tz="Europe/London").normalize()
tom   = today + pd.Timedelta(days=1)

def log_row(row):
    pd.DataFrame([row]).to_csv(LOG, mode="a",
        header=not os.path.exists(LOG), index=False)
    print("logged:", row)

def skip(reason):
    log_row({"date_made": today.date().isoformat(),
             "target_date": tom.date().isoformat(),
             "predicted_gw": "", "actual_gw": "", "error_gw": "",
             "status": f"skipped: {reason}"})

# ---- duplicate guard: don't forecast the same target_date twice ----
if os.path.exists(LOG):
    prev = pd.read_csv(LOG)
    if "target_date" in prev and (prev["target_date"] == tom.date().isoformat()).any():
        print("already have a row for", tom.date(), "- exiting.")
        raise SystemExit(0)

# ---- 1. fetch recent demand for the lags (the fragile part, wrapped) ----
try:
    end   = today.date()
    start = end - dt.timedelta(days=12)   # enough history for lag7 + buffer
    url = ("https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
           f"?settlementDateFrom={start}&settlementDateTo={end}&format=json")
    r = requests.get(url, timeout=60); r.raise_for_status()
    data = r.json().get("data", [])
    dem = pd.DataFrame(data)
    dem["settlementDate"] = pd.to_datetime(dem["settlementDate"]).dt.date
    dem["initialDemandOutturn"] = pd.to_numeric(dem["initialDemandOutturn"], errors="coerce")
    daily_peak = dem.groupby("settlementDate")["initialDemandOutturn"].max() / 1000.0  # GW

    # lag1 = the most recent COMPLETE day (yesterday); lag7 = 7 days before that
    d_lag1 = today.date() - dt.timedelta(days=1)
    d_lag7 = today.date() - dt.timedelta(days=7)
    lag1 = float(daily_peak.loc[d_lag1])
    lag7 = float(daily_peak.loc[d_lag7])
except Exception as e:
    skip(f"demand fetch failed ({type(e).__name__})")
    raise SystemExit(0)   # exit clean -> Action stays green, try again next run

# ---- 2. tomorrow's weather forecast (reliable API from v1) ----
try:
    wurl = ("https://api.open-meteo.com/v1/forecast"
            f"?latitude={LAT}&longitude={LON}"
            "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
            "&timezone=Europe%2FLondon&forecast_days=3")
    wd = requests.get(wurl, timeout=60).json()["daily"]
    wdf = pd.DataFrame(wd); wdf["time"] = pd.to_datetime(wdf["time"])
    w = wdf.loc[wdf["time"].dt.date == tom.date()].iloc[0]
    t_mean, t_max, t_min = (float(w["temperature_2m_mean"]),
                            float(w["temperature_2m_max"]),
                            float(w["temperature_2m_min"]))
except Exception as e:
    skip(f"weather fetch failed ({type(e).__name__})")
    raise SystemExit(0)

# ---- 3. features + predict ----
uk = holidays.country_holidays("GB", subdiv="ENG")
feat = {"t_mean":t_mean, "t_max":t_max, "t_min":t_min,
        "HDD":max(0.0, 15.5-t_mean), "CDD":max(0.0, t_mean-22.0),
        "dow":tom.dayofweek, "is_we":int(tom.dayofweek>=5),
        "month":tom.month, "doy":tom.dayofyear,
        "is_hol":int(tom.date() in uk), "lag1":lag1, "lag7":lag7}
X = pd.DataFrame([feat])[FEATURES]
booster = lgb.Booster(model_file="model_v2.txt")
pred = round(float(booster.predict(X)[0]), 2)

log_row({"date_made": today.date().isoformat(),
         "target_date": tom.date().isoformat(),
         "predicted_gw": pred, "actual_gw": "", "error_gw": "",
         "status": "ok"})
