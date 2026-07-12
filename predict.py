# predict.py - daily GB peak-demand forecast, appends one row to predictions_log.csv
import os
import pandas as pd
import requests
import lightgbm as lgb
import holidays

LAT, LON = 51.51, -0.13   # London as GB demand proxy
FEATURES = ["t_mean", "t_max", "t_min", "HDD", "CDD",
            "dow", "is_we", "month", "doy", "is_hol"]

today = pd.Timestamp.now(tz="Europe/London").normalize()
tom = today + pd.Timedelta(days=1)

url = ("https://api.open-meteo.com/v1/forecast"
       f"?latitude={LAT}&longitude={LON}"
       "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
       "&timezone=Europe%2FLondon&forecast_days=3")
wd = requests.get(url, timeout=60).json()["daily"]
wdf = pd.DataFrame(wd)
wdf["time"] = pd.to_datetime(wdf["time"])
w = wdf.loc[wdf["time"].dt.date == tom.date()].iloc[0]
t_mean, t_max, t_min = (float(w["temperature_2m_mean"]),
                        float(w["temperature_2m_max"]),
                        float(w["temperature_2m_min"]))

uk = holidays.country_holidays("GB", subdiv="ENG")
feat = {
    "t_mean": t_mean, "t_max": t_max, "t_min": t_min,
    "HDD": max(0.0, 15.5 - t_mean),
    "CDD": max(0.0, t_mean - 22.0),
    "dow": tom.dayofweek,
    "is_we": int(tom.dayofweek >= 5),
    "month": tom.month,
    "doy": tom.dayofyear,
    "is_hol": int(tom.date() in uk),
}
X = pd.DataFrame([feat])[FEATURES]

booster = lgb.Booster(model_file="model.txt")
pred = round(float(booster.predict(X)[0]), 2)

row = {"date_made": today.date().isoformat(),
       "target_date": tom.date().isoformat(),
       "predicted_gw": pred, "actual_gw": "", "error_gw": ""}
path = "predictions_log.csv"
pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
print("logged:", row)
