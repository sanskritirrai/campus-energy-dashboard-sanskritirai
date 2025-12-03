import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

p = Path("data")
files = list(p.glob("*.csv")) if p.exists() else []
df_list = []

for f in files:
    try:
        tmp = pd.read_csv(f, parse_dates=True, infer_datetime_format=True, on_bad_lines="skip")

        if "timestamp" in tmp.columns:
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
        elif "date" in tmp.columns:
            tmp["timestamp"] = pd.to_datetime(tmp["date"], errors="coerce")
        elif "datetime" in tmp.columns:
            tmp["timestamp"] = pd.to_datetime(tmp["datetime"], errors="coerce")
        else:
            tmp["timestamp"] = pd.to_datetime(tmp.iloc[:, 0], errors="coerce")

        if "kwh" not in tmp.columns:
            numcols = tmp.select_dtypes(include=["number"]).columns
            if len(numcols) > 0:
                tmp["kwh"] = tmp[numcols[0]]
            else:
                tmp["kwh"] = pd.to_numeric(tmp.iloc[:, 1], errors="coerce")

        tmp["kwh"] = pd.to_numeric(tmp["kwh"], errors="coerce")
        tmp = tmp.dropna(subset=["timestamp"])

        name = f.stem
        tmp["building"] = name
        tmp["month"] = tmp["timestamp"].dt.to_period("M").astype(str)

        df_list.append(tmp[["timestamp", "kwh", "building", "month"]])

    except FileNotFoundError:
        logging.error(f"{f} not found")
    except Exception as e:
        logging.error(f"error reading {f}: {e}")

if len(df_list) == 0:
    df = pd.DataFrame(columns=["timestamp", "kwh", "building", "month"])
else:
    df = pd.concat(df_list, ignore_index=True)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp")
df = df.reset_index(drop=True)
df_combined = df.copy()


def calculate_daily_totals(d):
    return d.set_index("timestamp").groupby("building")["kwh"].resample("D").sum().reset_index()


def calculate_weekly_aggregates(d):
    return d.set_index("timestamp").groupby("building")["kwh"].resample("W").sum().reset_index()


def building_wise_summary(d):
    g = d.groupby("building")["kwh"]
    s = g.agg(["mean", "min", "max", "sum"]).reset_index()
    s.columns = ["building", "mean_kwh", "min_kwh", "max_kwh", "total_kwh"]

    dct = {}
    for _, row in s.iterrows():
        dct[row["building"]] = {
            "mean": row["mean_kwh"],
            "min": row["min_kwh"],
            "max": row["max_kwh"],
            "total": row["total_kwh"]
        }

    return dct, s


df_daily = calculate_daily_totals(df_combined)
df_weekly = calculate_weekly_aggregates(df_combined)
summary_dict, df_summary = building_wise_summary(df_combined)


class MeterReading:
    def __init__(self, t, k):
        self.timestamp = pd.to_datetime(t)
        self.kwh = float(k)


class Building:
    def __init__(self, n):
        self.name = n
        self.readings = []

    def add_reading(self, r):
        self.readings.append(r)

    def calculate_total_consumption(self):
        return sum(r.kwh for r in self.readings)

    def generate_report(self):
        if not self.readings:
            return pd.DataFrame(columns=["timestamp", "kwh"])
        arr = [{"timestamp": r.timestamp, "kwh": r.kwh} for r in self.readings]
        df = pd.DataFrame(arr)
        df = df.sort_values("timestamp")
        return df


class BuildingManager:
    def __init__(self):
        self.buildings = {}

    def add_from_df(self, d):
        for _, r in d.iterrows():
            b = r["building"]
            if b not in self.buildings:
                self.buildings[b] = Building(b)
            self.buildings[b].add_reading(MeterReading(r["timestamp"], r["kwh"]))

    def totals(self):
        return {b: self.buildings[b].calculate_total_consumption() for b in self.buildings}

    def reports(self):
        return {b: self.buildings[b].generate_report() for b in self.buildings}


mgr = BuildingManager()
mgr.add_from_df(df_combined)
totals = mgr.totals()

if not os.path.exists("output"):
    os.makedirs("output")

df_combined.to_csv("output/cleaned_energy_data.csv", index=False)
df_summary.to_csv("output/building_summary.csv", index=False)

total_campus = df_combined["kwh"].sum()

if df_summary.shape[0] > 0:
    top_building = df_summary.sort_values("total_kwh", ascending=False).iloc[0]["building"]
else:
    top_building = None

if not df_combined.empty:
    df_hour = df_combined.set_index("timestamp").groupby("building")["kwh"].resample("H").sum().reset_index()
    peak_row = df_hour.loc[df_hour["kwh"].idxmax()] if not df_hour.empty else None
    peak_time = peak_row["timestamp"] if peak_row is not None else None
else:
    peak_time = None

with open("output/summary.txt", "w") as f:
    f.write(f"total_campus_consumption,{total_campus}\n")
    f.write(f"highest_consuming_building,{top_building}\n")
    f.write(f"peak_load_time,{peak_time}\n")
    f.write("weekly_and_daily_trends_described_in_csv_files\n")


# --------------------------------------------------
# FIXED PLOTTING SECTION (NO WARNINGS)
# --------------------------------------------------

plt.figure(figsize=(12,9))

# DAILY TREND
ax1 = plt.subplot(3,1,1)
for b in df_combined["building"].unique():
    tmp = df_combined[df_combined["building"] == b].set_index("timestamp").resample("D")["kwh"].sum()
    ax1.plot(tmp.index, tmp.values, label=str(b))
ax1.set_title("Daily consumption trend")
ax1.set_xlabel("Date")
ax1.set_ylabel("kWh")
if len(df_combined["building"].unique()) > 0:
    ax1.legend()

# WEEKLY AVERAGE
ax2 = plt.subplot(3,1,2)
t = df_combined.copy()
t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
wk = t.groupby(["building", pd.Grouper(key="timestamp", freq="W")])["kwh"].mean().reset_index()
avgwk = wk.groupby("building")["kwh"].mean().reset_index()

ax2.bar(avgwk["building"].astype(str), avgwk["kwh"])
ax2.set_title("Average weekly usage by building")
ax2.set_xlabel("Building")
ax2.set_ylabel("Avg weekly kWh")

# PEAK HOUR SCATTER
ax3 = plt.subplot(3,1,3)
if not df_combined.empty:
    hr = df_combined.set_index("timestamp").groupby("building")["kwh"].resample("H").sum().reset_index()
    peak = hr.sort_values("kwh", ascending=False).head(200)

    for b in peak["building"].unique():
        sub = peak[peak["building"] == b]
        ax3.scatter(sub["timestamp"], sub["kwh"], label=str(b), s=10)

ax3.set_title("Peak-hour consumption scatter")
ax3.set_xlabel("Time")
ax3.set_ylabel("kWh")
if not df_combined.empty:
    ax3.legend()

plt.tight_layout()
plt.savefig("output/dashboard.png")
plt.close()

print("files saved: output/cleaned_energy_data.csv, output/building_summary.csv, output/summary.txt, output/dashboard.png")
