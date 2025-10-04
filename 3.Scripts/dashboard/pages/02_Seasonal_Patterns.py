# nyc_citibike_dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="NYC CitiBike 2022 Analysis", layout="wide")

st.title("NYC CitiBike 2022 Analysis")

st.markdown("""
## Introduction (Part 2 of 2): Getting Acquainted with the 2022 Citibike Dataset
""")

st.markdown("""
### Chart 6: How Do Trip Volume and Imbalance Ratios Change Through the Seasons Across NYC Boroughs?

**The chart shows:**
- **Average Trips**: How many rides typically start/end in each neighborhood per hour.
- **Imbalance Ratio**: (starts - ends) / (starts + ends).  
    - Near **-1** → many more ends → docks may fill up.  
    - Near **+1** → many more starts → bikes may run out.  
    - Near **0** → balanced flow.  
- Views: *Average Day* (typical 24-hour pattern) or *Seasonal* (winter, spring, summer, fall)
- Filters: Select **boroughs** and **neighborhoods (NTAs)** to focus on specific areas.

As expected, we see changes in trip volume through the seasons. More interestingly, we also observe 
imbalance ratios shifting through the seasons for different boroughs and neighborhoods, revealing 
distinct usage patterns across NYC's geography and climate variations.
""")


# --- Load datasets from same folder as this script ---
BASE = Path(r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data")
PATH_DAILY_TRIPS  = BASE / "nta_daily_profile.pkl"         # expects: nta_name, borough, hour_of_day, avg_total_trips
PATH_SEASON_TRIPS = BASE / "nta_seasonal_profile.pkl"      # expects: nta_name, borough, season, hour_of_day, avg_total_trips
PATH_DAILY_IMB    = BASE / "nta_daily_imbalance.pkl"       # expects: nta_name, borough, hour_of_day, imbalance_ratio
PATH_SEASON_IMB   = BASE / "nta_seasonal_imbalance.pkl"    # expects: nta_name, borough, season, hour_of_day, imbalance_ratio

# quick sanity display
for p in [PATH_DAILY_TRIPS, PATH_SEASON_TRIPS, PATH_DAILY_IMB, PATH_SEASON_IMB]:
    st.sidebar.caption(f"{p.name}: exists={p.exists()} · {p.resolve()}")

@st.cache_data
def load_pkl(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p.resolve()}")
    return pd.read_pickle(p)

daily_trips   = load_pkl(PATH_DAILY_TRIPS)
season_trips  = load_pkl(PATH_SEASON_TRIPS)
daily_imb     = load_pkl(PATH_DAILY_IMB)
season_imb    = load_pkl(PATH_SEASON_IMB)

# --- Sidebar controls ---
mode   = st.sidebar.radio("View mode", ["Average Day", "Seasonal"], index=0, key="mode")
metric = st.sidebar.radio("Metric", ["Avg Trips", "Imbalance"], index=0, key="metric")

# Select working dataframe + value column based on controls
if mode == "Average Day":
    df_all = daily_trips if metric == "Avg Trips" else daily_imb
    value_col = "avg_total_trips" if metric == "Avg Trips" else "imbalance_ratio"
else:
    df_all = season_trips if metric == "Avg Trips" else season_imb
    value_col = "avg_total_trips" if metric == "Avg Trips" else "imbalance_ratio"

# Seasonal filter (if applicable)
if mode == "Seasonal":
    seasons = ["winter", "spring", "summer", "fall"]
    sel_season = st.sidebar.selectbox("Season", seasons, index=0, key="season")
    df = df_all[df_all.get("season", sel_season) == sel_season]
else:
    df = df_all.copy()

# Borough and NTA filters
boroughs = sorted(df["borough"].dropna().unique())
sel_borough = st.sidebar.selectbox("Borough", ["All"] + boroughs, index=0, key=f"borough_{mode}_{metric}")
if sel_borough != "All":
    df = df[df["borough"] == sel_borough]

ntas_all = sorted(df["nta_name"].dropna().unique())
top_default = (
    df.groupby("nta_name")[value_col].mean()
      .sort_values(ascending=False if metric == "Avg Trips" else True)
      .head(12).index.tolist()
)
sel_ntas = st.sidebar.multiselect(
    "Neighborhoods (NTA)",
    ntas_all,
    default=[n for n in top_default if n in ntas_all],
    key=f"ntas_{mode}_{metric}"
)
sel_ntas = [n for n in sel_ntas if n in ntas_all]
if not sel_ntas:
    st.info("Select at least one neighborhood."); st.stop()

# --- Plot ---
plot_df = (
    df[df["nta_name"].isin(sel_ntas)]
      .pivot_table(index="hour_of_day", columns="nta_name", values=value_col)
      .sort_index()
)

fig, ax = plt.subplots(figsize=(10, 6))
plot_df.plot(ax=ax)
ylabel = "Avg Total Trips / Hour" if metric == "Avg Trips" else "Imbalance (−1 to +1)"
ax.set_xlabel("Hour of Day"); ax.set_ylabel(ylabel)

title_scope  = "All Boroughs" if sel_borough == "All" else sel_borough
title_season = f" — {sel_season.capitalize()}" if mode == "Seasonal" else ""
ax.set_title(f"{mode}{title_season} — {metric} ({title_scope})")

if metric == "Imbalance":
    ax.set_ylim(-1, 1)
    ax.axhline(0, linestyle="--", linewidth=1)

ax.legend(title="NTA", ncol=2, fontsize=8)
st.pyplot(fig)
if metric == "Imbalance":
    st.caption("Imbalance: -1 = more **ends** (docks may fill up), +1 = more **starts** (bikes may run out).")

st.markdown("---")
st.markdown("""
Now that we are acquainted with our dataset, we can proceed to explore three operational questions. 
From two of them, we will derive actionable recommendations for CitiBike's operations!
""")