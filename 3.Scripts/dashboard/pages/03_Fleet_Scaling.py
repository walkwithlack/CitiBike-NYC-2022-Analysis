# Q1 — Fleet Scaling (repo-relative paths; works on Streamlit Cloud)

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
from _paths import csv_path 

st.set_page_config(page_title="Fleet Scaling Analysis", page_icon="📊", layout="wide")

st.title("Q1: How much should we scale bikes back between November and April?")
st.markdown(
    "We know already that weather affects demand patterns overall, but this does not determine the operational ceiling "
    "we need to plan for. For scaling decisions, what matters is **peak demand at busy hours (and not total volume or daily averages)**. "
    "We group our rides by month and hour of day to find the maximum hourly trips in each month. Then we can compare winter peaks as a "
    "percentage of summer peaks, and by applying a margin of 10–15%, we can answer the question of how much we could scale back while "
    "still covering demand safely. "
    "*Note*: This chart was built on preprocessed data. To check their creation see NYC_Q1_Scaling_back.ipynb."
)

# ──────────────────────────────────────────────────────────────────────────────
# Load data (from repo)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_peaks() -> pd.DataFrame:
    df = pd.read_csv(csv_path("peaks.csv")).copy()
    if "trips" not in df.columns:
        raise KeyError("Column 'trips' is required in peaks.csv")

    # month label
    if "month_label" not in df.columns:
        if "month" in df.columns:
            month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                         7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            df["month_label"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64").map(month_map)
        df["month_label"] = df["month_label"].fillna(
            df.get("month", pd.Series([None]*len(df))).astype(str).str.slice(0,3).str.title()
        )
        df["month_label"] = df["month_label"].fillna(pd.Series(range(1, len(df)+1), dtype="Int64").astype(str))

    # peak hour string
    if "peak_hour_str" not in df.columns:
        if "peak_hour" in df.columns:
            h = pd.to_numeric(df["peak_hour"], errors="coerce").fillna(0).astype(int).clip(0, 23)
            df["peak_hour_str"] = h.astype(str).str.zfill(2) + ":00–" + ((h + 1) % 24).astype(str).str.zfill(2) + ":00"
        else:
            df["peak_hour_str"] = ""

    return df

try:
    peaks = load_peaks()
except Exception as e:
    st.error(str(e))
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Ensure % of September peak
# ──────────────────────────────────────────────────────────────────────────────
pct_col = "%_of_september_peak"
have_pct = pct_col in peaks.columns

if not have_pct:
    with st.sidebar:
        st.markdown("**Percentage baseline** (only needed if CSV lacks '%_of_september_peak')")
        sept_trips = st.number_input(
            "September peak-hour trips (baseline)",
            min_value=1, step=1, value=50000,
            help="Used only to compute % of September peak when the CSV doesn't provide it."
        )
    peaks[pct_col] = (peaks["trips"] / float(sept_trips) * 100).round(1)

# ──────────────────────────────────────────────────────────────────────────────
# Plot (interactive)
# ──────────────────────────────────────────────────────────────────────────────
customdata_cols = ["peak_hour_str"]
hover_template = (
    "<b>%{x} 2022</b>"
    "<br>Peak hour: %{customdata[0]}"
    "<br>Peak-hour trips: %{y:,}"
)
if pct_col in peaks.columns:
    customdata_cols.append(pct_col)
    hover_template += "<br>% of Sept peak: %{customdata[1]:.1f}%"

# Exact Flare colorscale from seaborn
palette = sns.color_palette("flare", 11)
def rgb_tuple_to_str(t):
    r, g, b = (int(round(255*x)) for x in t)
    return f"rgb({r},{g},{b})"
colorscale = [[i/(len(palette)-1), rgb_tuple_to_str(c)] for i, c in enumerate(palette)]

fig = px.bar(
    peaks,
    x="month_label",
    y="trips",
    color="trips",
    color_continuous_scale=colorscale,
    labels={"month_label": "Month", "trips": "Peak-hour trips"},
    title="Monthly Peak-Hour — Citi Bike NYC (2022)"
)

fig.update_traces(
    hovertemplate=hover_template + "<extra></extra>",
    customdata=peaks[customdata_cols].to_numpy()
)

# Baseline at September's peak
if have_pct:
    if (peaks[pct_col] == 100).any():
        sept_peak_line = float(peaks.loc[peaks[pct_col].idxmax(), "trips"])
    else:
        sept_peak_line = float((peaks["trips"] / (peaks[pct_col] / 100.0)).max())
else:
    sept_peak_line = float(sept_trips)

fig.add_hline(
    y=sept_peak_line,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Sept peak = {int(sept_peak_line):,} trips",
    annotation_position="top left"
)

fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Peak-hour trips",
    xaxis_tickangle=-45,
    margin=dict(l=40, r=20, t=60, b=60),
    coloraxis_colorbar_title="Trips"
)

st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Recommendations
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Scaling Recommendations")
st.markdown("""
Based on peak-hour demand with a 10–15% safety margin:
- **Nov**: ~88% → no major cut  
- **Dec**: ~55% → **scale back 30–40%**  
- **Jan**: <40% → **scale back ~50%**  
- **Feb**: ~55% → **~70% capacity**  
- **Mar**: 70%+ → **80–85% capacity**  
- **Apr**: ~90% → **full capacity**
""")

# ──────────────────────────────────────────────────────────────────────────────
# Data preview & download
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Data used in this chart")
show_cols = ["month_label", "peak_hour_str", "trips", pct_col]
show_cols = [c for c in show_cols if c in peaks.columns]
st.dataframe(peaks[show_cols])

csv = peaks[show_cols].to_csv(index=False)
st.download_button(
    "Download monthly peaks (CSV)",
    data=csv,
    file_name="citibike_monthly_peaks_2022.csv",
    mime="text/csv"
)
