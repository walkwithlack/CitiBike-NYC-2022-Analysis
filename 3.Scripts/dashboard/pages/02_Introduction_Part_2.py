# Q0 — Introduction (Part 2): Borough & NTA seasonal patterns 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from _paths import csv_path 

st.set_page_config(page_title="Introduction (Part 2 of 2): Getting Acquainted with the 2022 Citi Bike Dataset", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def read_pickle(name: str) -> pd.DataFrame:
    # name is a file like "nta_daily_profile.pkl"
    return pd.read_pickle(csv_path(name))

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("Introduction (Part 2 of 2): Getting Acquainted with the 2022 Citi Bike Dataset")

# ──────────────────────────────────────────────────────────────────────────────
# Chart 5 — Borough/Neighborhood seasonality (Trips & Imbalance)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
### Chart 5: How Do Trip Volume and Imbalance Ratios Change Through the Seasons Across NYC Boroughs?

*Note*: This chart was built on pre-processed files. To check how these files were created, see NYC_02_Intro_Seasonal_Multi-line_Chart.ipynb.

**The chart shows:**
- **Average Trips**: How many rides typically start/end in each neighborhood per hour.
- **Imbalance Ratio**: (starts − ends) / (starts + ends)
  - Near **−1** → many more ends → docks may fill up
  - Near **+1** → many more starts → bikes may run out
  - Near **0** → balanced flow
- Views: *Average Day* (typical 24-hour pattern) or *Seasonal* (winter, spring, summer, fall)
- Filters: Select **boroughs** and **neighborhoods (NTAs)**.
""")

# Load data (repo-relative: no C:\ paths)
try:
    daily_trips  = read_pickle("nta_daily_profile.pkl")    # cols: nta_name, borough, hour_of_day, avg_total_trips
    season_trips = read_pickle("nta_seasonal_profile.pkl") # + season
    daily_imb    = read_pickle("nta_daily_imbalance.pkl")  # cols: nta_name, borough, hour_of_day, imbalance_ratio
    season_imb   = read_pickle("nta_seasonal_imbalance.pkl") # + season
except Exception as e:
    st.error(f"Failed to load one or more PKL files.\n\n{e}")
    st.stop()

# Controls
mode   = st.sidebar.radio("View mode", ["Average Day", "Seasonal"], index=0, key="mode")
metric = st.sidebar.radio("Metric", ["Avg Trips", "Imbalance"], index=0, key="metric")

if mode == "Average Day":
    df_all = daily_trips if metric == "Avg Trips" else daily_imb
    value_col = "avg_total_trips" if metric == "Avg Trips" else "imbalance_ratio"
else:
    df_all = season_trips if metric == "Avg Trips" else season_imb
    value_col = "avg_total_trips" if metric == "Avg Trips" else "imbalance_ratio"

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

# Plot
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
    st.caption("Imbalance: −1 = more **ends** (docks may fill up), +1 = more **starts** (bikes may run out).")

st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# Chart 6 — Kepler.gl Map of Largest Flows (Origin–Destination)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Chart 6: The Citi Bike Network in Action")
st.markdown("*Map showing highest-volume stations and origin–destination flows of 1500+ trips*")
st.markdown("*Note*: Built from pre-processed data. See NYC_2.5_Kepler.gl_Preprocessing.ipynb & make_kepler_map_light.py")

import streamlit.components.v1 as components
from _paths import REPO_ROOT  # reuse repo root

html_path = REPO_ROOT / "3.Scripts" / "dashboard" / "assets" / "kepler_map_light.html"
if html_path.exists():
    components.html(html_path.read_text(encoding="utf-8"), height=650, scrolling=False)
else:
    st.error(f"Map HTML not found at: {html_path}")
    st.caption("Make sure the lightweight HTML is committed to the repo.")


st.markdown("""
We see some of the strongest flows along the Hudson waterfront and through Central Park — indicating that **even though Citi Bike mainly supports short first- or last-mile trips, its most popular routes are leisurely rides**.
""")

st.markdown("""
Now that we are better acquainted with our dataset, we can proceed to explore **three operational questions**. 
**From two of them**, we will derive **actionable recommendations for Citi Bike's operations**!
""")
