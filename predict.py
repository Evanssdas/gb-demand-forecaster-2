# predict.py - self-grading GB peak-demand forecaster (v2, lag-aware, auto-actuals).
# Each run: (1) backfills actuals for past-due predictions, (2) predicts tomorrow.
# Weather+calendar+lag model; graceful skip on fetch failure; duplicate-guarded.
import os, datetime as dt
import numpy as np, pandas as pd, requests
import lightgbm as lgb, holidays

LAT, LON = 51.51, -0.13
LOG = "predictions_log.csv"
FEATURES = ["t_mean","t_max","t_min","HDD","CDD","dow","is_we","month","doy","is_hol","lag1","lag7"]
COLS = ["date_made","target_date","predicted_gw","actual_gw","error_gw","status"]

today = pd.Timestamp.now(tz="Europe/London").normalize()
tom   = today + pd.Timedelta(days=1)

log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=COLS)
for c in COLS:                       # tolerate an older log missing the status column
    if c not in log.columns: log[c] = ""

def peak_demand_gw(start, end):
    """daily PEAK national demand (GW) from Elexon, for both actuals and lags."""
    rows=[]
    d=start
    while d<=end:
        d2=min(d+dt.timedelta(days=6), end)
        u=("https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
           f"?settlementDateFrom={d}&settlementDateTo={d2}&format=json")
        rows+=requests.get(u,timeout=60).json().get("data",[])
        d=d2+dt.timedelta(days=1)
    df=pd.DataFrame(rows)
    df["settlementDate"]=pd.to_datetime(df["settlementDate"]).dt.date
    df["initialDemandOutturn"]=pd.to_numeric(df["initialDemandOutturn"],errors="coerce")
    return df.groupby("settlementDate")["initialDemandOutturn"].max()/1000.0

# ---------- 1. BACKFILL actuals for past-due rows ----------
try:
    due = log[(log["actual_gw"].isna() | (log["actual_gw"].astype(str).str.strip()=="")) &
              (pd.to_datetime(log["target_date"]).dt.date < today.date())]
    if len(due):
        lo=pd.to_datetime(due["target_date"]).min().date()
        hi=pd.to_datetime(due["target_date"]).max().date()
        act=peak_demand_gw(lo, hi)
        for i,row in due.iterrows():
            d=pd.to_datetime(row["target_date"]).date()
            if d in act.index:
                a=round(float(act.loc[d]),2)
                log.at[i,"actual_gw"]=a
                try: log.at[i,"error_gw"]=round(float(row["predicted_gw"])-a,2)
                except (ValueError,TypeError): pass
        print("actuals backfilled where available")
except Exception as e:
    print("backfill skipped:", type(e).__name__)

# ---------- 2. PREDICT tomorrow (skip if already present) ----------
if not (log["target_date"].astype(str)==tom.date().isoformat()).any():
    try:
        # lags: recent actual peaks (yesterday and a week ago)
        recent=peak_demand_gw(today.date()-dt.timedelta(days=9), today.date()-dt.timedelta(days=1))
        lag1=float(recent.loc[today.date()-dt.timedelta(days=1)])
        lag7=float(recent.loc[today.date()-dt.timedelta(days=7)])
        # tomorrow's weather
        wu=("https://api.open-meteo.com/v1/forecast"
            f"?latitude={LAT}&longitude={LON}"
            "&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
            "&timezone=Europe%2FLondon&forecast_days=3")
        wdf=pd.DataFrame(requests.get(wu,timeout=60).json()["daily"]); wdf["time"]=pd.to_datetime(wdf["time"])
        w=wdf.loc[wdf["time"].dt.date==tom.date()].iloc[0]
        t_mean,t_max,t_min=float(w["temperature_2m_mean"]),float(w["temperature_2m_max"]),float(w["temperature_2m_min"])
        uk=holidays.country_holidays("GB", subdiv="ENG")
        feat={"t_mean":t_mean,"t_max":t_max,"t_min":t_min,
              "HDD":max(0.0,15.5-t_mean),"CDD":max(0.0,t_mean-22.0),
              "dow":tom.dayofweek,"is_we":int(tom.dayofweek>=5),"month":tom.month,"doy":tom.dayofyear,
              "is_hol":int(tom.date() in uk),"lag1":lag1,"lag7":lag7}
        booster=lgb.Booster(model_file="model_v2.txt")
        pred=round(float(booster.predict(pd.DataFrame([feat])[FEATURES])[0]),2)
        new={"date_made":today.date().isoformat(),"target_date":tom.date().isoformat(),
             "predicted_gw":pred,"actual_gw":"","error_gw":"","status":"ok"}
    except Exception as e:
        new={"date_made":today.date().isoformat(),"target_date":tom.date().isoformat(),
             "predicted_gw":"","actual_gw":"","error_gw":"","status":f"skipped: {type(e).__name__}"}
    log=pd.concat([log, pd.DataFrame([new])], ignore_index=True)
    print("logged:", new)
else:
    print("already have", tom.date())

log[COLS].to_csv(LOG, index=False)
