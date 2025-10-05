# Q0 — Introduction (Part 2): Borough & NTA seasonal patterns
# Works with local paths OR Google Drive links/IDs for the four PKL files.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
import requests
import re

st.set_page_config(page_title="NYC CitiBike 2022 Analysis", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: accept local paths OR Google Drive links/IDs and load PKL safely
# ──────────────────────────────────────────────────────────────────────────────

def gdrive_to_direct(url_or_id: str) -> str:
    """Return a direct-download URL for a Google Drive share link or raw file ID."""
    s = (url_or_id or "").strip()
    # raw file ID?
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        return f"https://drive.google.com/uc?export=download&id={s}"
    # already direct?
    if "uc?export=download&id=" in s:
        return s
    # common share formats
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", s) or re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", s)
    return f"https://drive.google.com/uc?export=download&id={m.group(1)}" if m else s

def download_url(url: str) -> BytesIO:
    """Stream a file; handles Google Drive large-file confirm flow. Returns BytesIO."""
    with requests.Session() as sess:
        r = sess.get(url, stream=True)
        # for Drive: fetch confirm token if present
        if "drive.google.com" in url and "confirm=" not in r.url:
            for k, v in r.cookies.items():
                if k.startswith("download_warning"):
                    url2 = url + ("&" if "?" in url else "?") + f"confirm={v}"
                    r = sess.get(url2, stream=True)
                    break
        r.raise_for_status()
        buf = BytesIO()
        for chunk in r.iter_content(2 * 1024 * 1024):
            if chunk:
                buf.write(chunk)
        buf.seek(0)
        return buf

@st.cache_data(show_spinner=False)
def read_pickle_safely(user_input: str):
    """
    If 'user_input' is a local file and exists → read from disk.
    Else treat it as a URL / Drive link / Drive ID → download and read.
    """
    p = Path(user_input)
    if p.exists():
        return pd.read_pickle(p)
    # download from Drive/URL
    url = gdrive_to_direct(user_input)
    with st.spinner("Downloading data from Google Drive…"):
        buf = download_url(url)
    return pd.read_pickle(buf)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

st.title("NYC CitiBike 2022 Analysis")
st.markdown("## Introduction (Part 2 of 2): Getting Acquainted with the 2022 Citibike Dataset")

st.markdown("""
### Chart 6: How Do Trip Volume and Imbalance Ratios Change Through the Seasons Across NYC Boroughs?

**The chart shows:**
- **Average Trips**: How many rides typically start/end in each neighborhood per hour.
- **Imbalance Ratio**: (starts - ends) / (starts + ends)  
  - Near **-1** → many more ends → docks may fill up  
  - Near **+1** → many more starts → bikes may run out  
  - Near **0** → balanced flow  
- Views: *Average Day* (typical 24-hour pattern) or *Seasonal* (winter, spring, summer, fall)
- Filters: Select **boroughs** and **neighborhoods (NTAs)**.

We see seasonal volume shifts and, more interestingly, **imbalance** shifts by borough/nta.
""")

# ──────────────────────────────────────────────────────────────────────────────
# Inputs (local path OR Drive link/ID). Defaults set to your Drive file IDs.
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Data (local path or Google Drive link/ID)")

PATH_DAILY_TRIPS  = st.sidebar.text_input(
    "nta_daily_profile.pkl",
    value="1FOHo5ZTasdYrZhBvHjoM2KF00u9882fF",
    help="Local path or Google Drive link/ID",
)
PATH_SEASON_TRIPS = st.sidebar.text_input(
    "nta_seasonal_profile.pkl",
    value="1awGMMQv_7KLijqMFWlkgT4FhauH8cMH9",
    help="Local path or Google Drive link/ID",
)
PATH_DAILY_IMB    = st.sidebar.text_input(
    "nta_daily_imbalance.pkl",
    value="1_Ckv3_MlOEL4_dxz6LrURoB-w0yKf3lB",
    help="Local path or Google Drive link/ID",
)
PATH_SEASON_IMB   = st.sidebar.text_input(
    "nta_seasonal_imbalance.pkl",
    value="1kMq6t3_ftO6kMjax6segbTcoMvHAmrtX",
    help="Local path or Google Drive link/ID",
)

# quick sanity display (optional)
for label, p in [
    ("Daily Trips", PATH_DAILY_TRIPS),
    ("Seasonal Trips", PATH_SEASON_TRIPS),
    ("Daily Imbalance", PATH_DAILY_IMB),
    ("Seasonal Imbalance", PATH_SEASON_IMB),
]:
    st.sidebar.caption(f"{label}: {p}")

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

try:
    daily_trips   = read_pickle_safely(PATH_DAILY_TRIPS)      # cols: nta_name, borough, hour_of_day, avg_total_trips
    season_trips  = read_pickle_safely(PATH_SEASON_TRIPS)     # + season
    daily_imb     = read_pickle_safely(PATH_DAILY_IMB)        # cols: nta_name, borough, hour_of_day, imbalance_ratio
    season_imb    = read_pickle_safely(PATH_SEASON_IMB)       # + season
except Exception as e:
    st.error(f"Failed to load one or more PKL files.\n\n{e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Controls
# ──────────────────────────────────────────────────────────────────────────────

mode   = st.sidebar.radio("View mode", ["Average Day", "Seasonal"], index=0, key="mode")
metric = st.sidebar.radio("Metric", ["Avg Trips", "Imbalance"], index=0, key="metric")

if mode == "Average Day":
    df_all = daily_trips if metric == "Avg Trips" else daily_imb
    value_col = "avg_total_trips" if metric == "Avg Trips" else "imbalance_ratio"
else:
    df_all = season_trips if metric == "Avg Trips" else season_imb
    value_col = "avg_total_trips" if metric == "Avg Trips" else "imbalance_ratio"

# Optional season filter
if mode == "Seasonal" and "season" in df_all.columns:
    seasons = ["winter", "spring", "summer", "fall"]
    sel_season = st.sidebar.selectbox("Season", seasons, index=0, key="season")
    df = df_all[df_all["season"] == sel_season].copy()
else:
    df = df_all.copy()
    sel_season = None

# Borough & NTA filters
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
    st.info("Select at least one neighborhood.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

plot_df = (
    df[df["nta_name"].isin(sel_ntas)]
      .pivot_table(index="hour_of_day", columns="nta_name", values=value_col)
      .sort_index()
)

fig, ax = plt.subplots(figsize=(10, 6))
plot_df.plot(ax=ax)
ylabel = "Avg Total Trips / Hour" if metric == "Avg Trips" else "Imbalance (−1 to +1)"
ax.set_xlabel("Hour of Day")
ax.set_ylabel(ylabel)

title_scope  = "All Boroughs" if sel_borough == "All" else sel_borough
title_season = f" — {sel_season.capitalize()}" if sel_season else ""
ax.set_title(f"{mode}{title_season} — {metric} ({title_scope})")

if metric == "Imbalance":
    ax.set_ylim(-1, 1)
    ax.axhline(0, linestyle="--", linewidth=1, color="#666")

ax.legend(title="NTA", ncol=2, fontsize=8)
st.pyplot(fig)

if metric == "Imbalance":
    st.caption("Imbalance: -1 = more **ends** (docks may fill up), +1 = more **starts** (bikes may run out).")

st.markdown("---")
st.markdown("""
Now that we are acquainted with our dataset, we can proceed to explore three operational questions. 
From two of them, we will derive actionable recommendations for CitiBike's operations!
""")
