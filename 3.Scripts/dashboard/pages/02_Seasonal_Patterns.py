# nyc_citibike_dashboard.py (Page 2)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
import requests

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
""")

# --- Local data folder (your PC) ---
BASE = Path(r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data")
PATH_DAILY_TRIPS  = BASE / "nta_daily_profile.pkl"
PATH_SEASON_TRIPS = BASE / "nta_seasonal_profile.pkl"
PATH_DAILY_IMB    = BASE / "nta_daily_imbalance.pkl"
PATH_SEASON_IMB   = BASE / "nta_seasonal_imbalance.pkl"

# --- Google Drive fallback (for Streamlit Cloud) ---
# IDs supplied by you; #3 is pending (you sent a duplicate of #1).
DRIVE_FILES = {
    "nta_daily_profile.pkl":       "https://drive.google.com/file/d/1FOHo5ZTasdYrZhBvHjoM2KF00u9882fF/view?usp=sharing",
    "nta_seasonal_profile.pkl":    "https://drive.google.com/uc?export=download&id=1awGMMQv_7KLijqMFWlkgT4FhauH8cMH9",
    "nta_daily_imbalance.pkl":     "https://drive.google.com/uc?export=download&id=1_Ckv3_MlOEL4_dxz6LrURoB-w0yKf3lB",
    "nta_seasonal_imbalance.pkl":  "https://drive.google.com/uc?export=download&id=1kMq6t3_ftO6kMjax6segbTcoMvHAmrtX",
}

def _download_bytes(url: str) -> bytes:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        return b"".join(chunk for chunk in r.iter_content(chunk_size=262_144) if chunk)

@st.cache_data(show_spinner=False)
def load_pkl_any(p: Path):
    """Load a pickle from local path if present; else fetch from Google Drive mapping."""
    if p.exists():
        return pd.read_pickle(p)
    url = DRIVE_FILES.get(p.name)
    if not url or "REPLACE_WITH" in url:
        st.error(
            f"Missing **{p.name}** Google Drive ID. "
            "Please provide a valid shared link and update DRIVE_FILES."
        )
        st.stop()
    try:
        data = _download_bytes(url)
        return pd.read_pickle(BytesIO(data))
    except Exception as e:
        st.error(f"Failed to fetch **{p.name}** from Google Drive.\n\n{e}")
        st.stop()

# quick local existence display
for p in [PATH_DAILY_TRIPS, PATH_SEASON_TRIPS, PATH_DAILY_IMB, PATH_SEASON_IMB]:
    st.sidebar.caption(f"{p.name}: local_exists={p.exists()} · {p.resolve()}")

# --- Load data (local-or-Drive) ---
daily_trips   = load_pkl_any(PATH_DAILY_TRIPS)
season_trips  = load_pkl_any(PATH_SEASON_TRIPS)
daily_imb     = load_pkl_any(PATH_DAILY_IMB)
season_imb    = load_pkl_any(PATH_SEASON_IMB)

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
    df = df_all[df_all.get("season", sel_season) == sel_season] if "season" in df_all.columns else df_all.copy()
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
