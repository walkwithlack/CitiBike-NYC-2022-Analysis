import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Fleet Scaling Analysis", page_icon="📊", layout="wide")

# Title and introduction
st.title("Q1: How much should we scale bikes back between November and April?")

st.markdown("""
The dual-axis chart showing seasonal demand patterns reveals overall trends, but **operational 
planning requires understanding peak demand at busy hours**. For scaling decisions, what matters 
is the peak load per month, not daily or monthly averages.

We analyze monthly peak-hour demand to determine safe scaling recommendations with a 10-15% safety margin.
""")

# Data loading with caching
@st.cache_data
def load_hourly_data(file_path):
    """Load and aggregate trip data into hourly counts."""
    hourly_counts = {}
    
    for chunk in pd.read_csv(
        file_path,
        usecols=["started_at"],
        chunksize=500_000,
        dtype=str,
        on_bad_lines="skip",
        low_memory=True,
        encoding_errors="ignore"
    ):
        s = pd.to_datetime(chunk["started_at"], errors="coerce")
        s = s[(s >= "2022-01-01") & (s < "2023-01-01")]
        s_hour = s.dt.floor("H").dropna()
        
        vc = s_hour.value_counts()
        for ts, cnt in vc.items():
            hourly_counts[ts] = hourly_counts.get(ts, 0) + int(cnt)
    
    hourly = pd.Series(hourly_counts, dtype="int64").sort_index()
    return hourly

@st.cache_data
def calculate_monthly_peaks(hourly):
    """Calculate peak-hour demand for each month."""
    hourly_df = hourly.rename("trips").to_frame()
    hourly_df["month"] = hourly_df.index.to_period("M")
    
    # Find the hour with max trips in each month
    peak_idx = hourly_df.groupby("month")["trips"].idxmax()
    peaks = hourly_df.loc[peak_idx].copy()
    
    # Tidy up
    peaks = peaks.reset_index().rename(columns={"index": "peak_hour"})
    peaks["month"] = peaks["month"].astype(str)
    
    # Use September as baseline
    if (peaks["month"] == "2022-09").any():
        sept_peak = int(peaks.loc[peaks["month"] == "2022-09", "trips"].iloc[0])
    else:
        sept_peak = int(peaks["trips"].max())
    
    peaks["%_of_september_peak"] = (peaks["trips"] / sept_peak * 100).round(1)
    
    # Add friendly labels
    peaks["month_label"] = pd.to_datetime(peaks["month"] + "-01").dt.strftime("%b")
    peaks["peak_hour_str"] = pd.to_datetime(peaks["peak_hour"]).dt.strftime("%Y-%m-%d %H:%M")
    
    return peaks.sort_values("month"), sept_peak

# File path input
st.sidebar.header("Data Configuration")
default_path = r"C:/Users/magia/OneDrive/Desktop/NY_Citi_Bike/2.Data/Prepared Data/nyc_2022_essential_data.csv"
file_path = st.sidebar.text_input(
    "Data file path:",
    value=default_path,
    help="Path to your NYC CitiBike 2022 data CSV file"
)

# Load data
try:
    with st.spinner("Loading and processing data..."):
        hourly = load_hourly_data(file_path)
        peaks, sept_peak = calculate_monthly_peaks(hourly)
    
    st.success(f"✓ Processed {len(hourly):,} hours of data from 2022")
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("September Peak", f"{sept_peak:,} trips", help="Highest monthly peak-hour demand")
    with col2:
        min_peak = peaks["trips"].min()
        st.metric("January Peak", f"{min_peak:,} trips", 
                 delta=f"-{100 - (min_peak/sept_peak*100):.1f}%",
                 delta_color="inverse")
    with col3:
        avg_winter = peaks[peaks["month"].isin(["2022-12", "2023-01", "2023-02"])]["trips"].mean()
        st.metric("Winter Average Peak", f"{int(avg_winter):,} trips",
                 delta=f"-{100 - (avg_winter/sept_peak*100):.1f}%",
                 delta_color="inverse")
    
    st.markdown("---")
    
    # Create interactive visualization
    st.subheader("Monthly Peak-Hour Demand")
    
    # Flare colorscale (warm gradient)
    colorscale = [
        [0.0, "rgb(235,155,116)"],
        [0.1, "rgb(233,135,104)"],
        [0.2, "rgb(229,113,94)"],
        [0.3, "rgb(222,93,92)"],
        [0.4, "rgb(211,76,96)"],
        [0.5, "rgb(193,65,104)"],
        [0.6, "rgb(174,59,109)"],
        [0.7, "rgb(154,54,112)"],
        [0.8, "rgb(134,48,113)"],
        [0.9, "rgb(114,44,110)"],
        [1.0, "rgb(94,40,104)"]
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
        hovertemplate=(
            "<b>%{x} 2022</b>"
            "<br>Peak hour: %{customdata[0]}"
            "<br>Peak-hour trips: %{y:,}"
            "<br>% of Sept peak: %{customdata[1]:.1f}%<extra></extra>"
        ),
        customdata=np.c_[peaks["peak_hour_str"], peaks["%_of_september_peak"]],
    )
    
    # Add September baseline
    fig.add_hline(
        y=sept_peak,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Sept peak = {sept_peak:,} trips",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title="Monthly Peak-Hour Demand - CitiBike NYC 2022",
        xaxis_title="Month",
        yaxis_title="Peak-hour trips",
        xaxis_tickangle=-45,
        height=500,
        showlegend=False,
        coloraxis_colorbar_title="Trips"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed monthly breakdown
    st.markdown("---")
    st.subheader("Detailed Monthly Breakdown")
    
    display_df = peaks[["month_label", "peak_hour_str", "trips", "%_of_september_peak"]].copy()
    display_df.columns = ["Month", "Peak Hour", "Peak Trips", "% of Sept Peak"]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Peak Trips": st.column_config.NumberColumn(format="%d"),
            "% of Sept Peak": st.column_config.NumberColumn(format="%.1f%%")
        }
    )
    
    # Recommendations
    st.markdown("---")
    st.subheader("Scaling Recommendations")
    
    recommendations = """
    Based on peak-hour demand analysis with a 10-15% safety margin:
    
    - **November**: Peak demand remains strong (~88% of maximum). No significant scaling needed yet.
    
    - **December**: Sharp drop to ~55% of peak -> **Scale back fleet by 30-40%**
    
    - **January**: Demand falls below 40% of September peak -> **Scale back fleet by ~50%**
    
    - **February**: Recovery to ~55% -> **Increase to 70% capacity**
    
    - **March**: Continued growth to 70%+ -> **Increase to 80-85% capacity**
    
    - **April**: Approaching 90% of peak -> **Return to full fleet capacity**
    
    ### Key Insight
    
    A **month-by-month approach is more practical** than a blanket seasonal strategy. Winter peak 
    demand varies widely (from 37% to 88% of maximum), supporting targeted monthly scaling rather 
    than uniform reductions across the entire winter period.
    """
    
    st.markdown(recommendations)
    
    # Download option
    st.markdown("---")
    csv = peaks.to_csv(index=False)
    st.download_button(
        label="Download Monthly Peak Data (CSV)",
        data=csv,
        file_name="citibike_monthly_peaks_2022.csv",
        mime="text/csv"
    )

except FileNotFoundError:
    st.error(f"File not found: {file_path}")
    st.info("Please update the file path in the sidebar to point to your data file.")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please check your data file path and format.")