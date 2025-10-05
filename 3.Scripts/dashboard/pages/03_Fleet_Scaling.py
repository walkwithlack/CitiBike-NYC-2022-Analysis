# Q1 — Fleet Scaling (auto Google Drive or local path)

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
from io import BytesIO
import re
import tempfile
import requests

st.set_page_config(page_title="Fleet Scaling Analysis", page_icon="📊", layout="wide")

# -------- Helpers: resolve path or Drive link -> temp file --------

def _gdrive_direct_url(link: str) -> str | None:
    """Accept common Google Drive share formats and return a direct download URL, else None."""
    if not isinstance(link, str):
        return None
    m = re.search(r"/file/d/([a-zA-Z0-9_-]{20,})", link)
    if not m:
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]{20,})", link)
    if not m:
        if "drive.google.com/uc" in link and "id=" in link:
            return link
        return None
    fid = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={fid}"

@st.cache_data(show_spinner=False)
def _download_to_temp(url: str, suffix: str = ".csv") -> str:
    """Stream a remote file (incl. Google Drive confirm flow) to a temp file; return its path."""
    import tempfile, requests

    with requests.Session() as sess:
        r = sess.get(url, stream=True, timeout=300)
        # Handle Google Drive "virus scan too large" confirm step
        if "drive.google.com" in url and "confirm=" not in r.url:
            for k, v in r.cookies.items():
                if k.startswith("download_warning"):
                    sep = "&" if "?" in url else "?"
                    url = f"{url}{sep}confirm={v}"
                    r = sess.get(url, stream=True, timeout=300)
                    break
        r.raise_for_status()

        with tempfile.NamedTemporaryFile(prefix="citibike_", suffix=suffix, delete=False) as tmp:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):  # 4 MB chunks
                if chunk:
                    tmp.write(chunk)
            return tmp.name

def resolve_input_path(user_input: str) -> str:
    """
    If local path exists -> return it.
    If Drive link -> download once to temp and return path.
    Else -> raise with guidance.
    """
    p = Path(user_input)
    if p.exists():
        return str(p)

    gdrive_url = _gdrive_direct_url(user_input)
    if gdrive_url:
        st.info("Downloading data from Google Drive (cached after first download)...")
        return _download_to_temp(gdrive_url, suffix=".csv")

    raise FileNotFoundError(
        "Data file not found. Provide a valid local path or a Google Drive sharing link."
    )

# -------- Sidebar: default to your Drive link --------
st.sidebar.header("Data Configuration")
file_path_input = st.sidebar.text_input(
    "CSV path or Google Drive link:",
    value="https://drive.google.com/file/d/10kp8fxIR7h1McO-shf0I42cjtj6VViu-/view?usp=sharing",
    help="Paste a local path or a Google Drive share link to your 2022 trips CSV (essential subset)."
)

# -------- Page intro --------
st.title("Q1: How much should we scale bikes back between November and April?")

st.markdown("""
We care about **monthly peak-hour demand** (not averages) to size the fleet with a 10–15% safety margin.
""")

# -------- Data loading & transforms --------
@st.cache_data(show_spinner=True)
def load_hourly_data(resolved_path: str) -> pd.Series:
    """Aggregate hourly trip counts from CSV (chunked)."""
    hourly_counts: dict[pd.Timestamp, int] = {}
    for chunk in pd.read_csv(
        resolved_path,
        usecols=["started_at"],
        chunksize=500_000,
        dtype=str,
        on_bad_lines="skip",
        low_memory=True,
        encoding_errors="ignore",
    ):
        s = pd.to_datetime(chunk["started_at"], errors="coerce")
        s = s[(s >= "2022-01-01") & (s < "2023-01-01")]
        s_hour = s.dt.floor("H").dropna()
        vc = s_hour.value_counts()
        for ts, cnt in vc.items():
            hourly_counts[ts] = hourly_counts.get(ts, 0) + int(cnt)
    hourly = pd.Series(hourly_counts, dtype="int64").sort_index()
    return hourly

@st.cache_data(show_spinner=False)
def calculate_monthly_peaks(hourly: pd.Series):
    hourly_df = hourly.rename("trips").to_frame()
    hourly_df["month"] = hourly_df.index.to_period("M")
    peak_idx = hourly_df.groupby("month")["trips"].idxmax()
    peaks = hourly_df.loc[peak_idx].copy()
    peaks = peaks.reset_index().rename(columns={"index": "peak_hour"})
    peaks["month"] = peaks["month"].astype(str)
    if (peaks["month"] == "2022-09").any():
        sept_peak = int(peaks.loc[peaks["month"] == "2022-09", "trips"].iloc[0])
    else:
        sept_peak = int(peaks["trips"].max())
    peaks["%_of_september_peak"] = (peaks["trips"] / sept_peak * 100).round(1)
    peaks["month_label"] = pd.to_datetime(peaks["month"] + "-01").dt.strftime("%b")
    peaks["peak_hour_str"] = pd.to_datetime(peaks["peak_hour"]).dt.strftime("%Y-%m-%d %H:%M")
    return peaks.sort_values("month"), sept_peak

# -------- Run --------
try:
    resolved = resolve_input_path(file_path_input)
    with st.spinner("Loading and processing data..."):
        hourly = load_hourly_data(resolved)
        peaks, sept_peak = calculate_monthly_peaks(hourly)

    st.success(f"✓ Processed {len(hourly):,} hours of data from 2022")

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("September Peak", f"{sept_peak:,} trips", help="Highest monthly peak-hour demand")
    with col2:
        jan_val = (
            int(peaks.loc[peaks["month"] == "2022-01", "trips"].iloc[0])
            if (peaks["month"] == "2022-01").any()
            else peaks["trips"].min()
        )
        st.metric("January Peak", f"{jan_val:,} trips",
                  delta=f"-{100 - (jan_val / sept_peak * 100):.1f}%", delta_color="inverse")
    with col3:
        winter_mask = peaks["month"].isin(["2022-12", "2023-01", "2023-02"])
        avg_winter = peaks.loc[winter_mask, "trips"].mean() if winter_mask.any() else peaks["trips"].mean()
        st.metric("Winter Avg Peak", f"{int(avg_winter):,} trips",
                  delta=f"-{100 - (avg_winter / sept_peak * 100):.1f}%", delta_color="inverse")

    st.markdown("---")

    # Chart
    st.subheader("Monthly Peak-Hour Demand")
    colorscale = [
        [0.0, "rgb(235,155,116)"], [0.1, "rgb(233,135,104)"], [0.2, "rgb(229,113,94)"],
        [0.3, "rgb(222,93,92)"],  [0.4, "rgb(211,76,96)"],  [0.5, "rgb(193,65,104)"],
        [0.6, "rgb(174,59,109)"], [0.7, "rgb(154,54,112)"], [0.8, "rgb(134,48,113)"],
        [0.9, "rgb(114,44,110)"], [1.0, "rgb(94,40,104)"],
    ]

    fig = px.bar(
        peaks,
        x="month_label",
        y="trips",
        color="trips",
        color_continuous_scale=colorscale,
        labels={"month_label": "Month", "trips": "Peak-hour trips"},
    )
    fig.update_traces(
        hovertemplate="<b>%{x} 2022</b><br>Peak hour: %{customdata[0]}"
                      "<br>Peak-hour trips: %{y:,}"
                      "<br>% of Sept peak: %{customdata[1]:.1f}%<extra></extra>",
        customdata=np.c_[peaks["peak_hour_str"], peaks["%_of_september_peak"]],
    )
    fig.add_hline(y=sept_peak, line_dash="dash", line_color="red",
                  annotation_text=f"Sept peak = {sept_peak:,} trips", annotation_position="top left")
    fig.update_layout(
        title="Monthly Peak-Hour Demand — CitiBike NYC 2022",
        xaxis_title="Month", yaxis_title="Peak-hour trips",
        xaxis_tickangle=-45, height=500, showlegend=False, coloraxis_colorbar_title="Trips",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("---")
    st.subheader("Detailed Monthly Breakdown")
    display_df = peaks[["month_label", "peak_hour_str", "trips", "%_of_september_peak"]].copy()
    display_df.columns = ["Month", "Peak Hour", "Peak Trips", "% of Sept Peak"]
    st.dataframe(
        display_df, use_container_width=True, hide_index=True,
        column_config={"Peak Trips": st.column_config.NumberColumn(format="%d"),
                       "% of Sept Peak": st.column_config.NumberColumn(format="%.1f%%")}
    )

    # Recommendations
    st.markdown("---")
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

    # Download
    st.markdown("---")
    csv = peaks.to_csv(index=False)
    st.download_button("Download Monthly Peak Data (CSV)", data=csv,
                       file_name="citibike_monthly_peaks_2022.csv", mime="text/csv")

except FileNotFoundError as e:
    st.error(str(e))
    st.info("Tip: paste a Google Drive sharing link in the sidebar — it will download & cache automatically.")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please check your data path/link and format.")
