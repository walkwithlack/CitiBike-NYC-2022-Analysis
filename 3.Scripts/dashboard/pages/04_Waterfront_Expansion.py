import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Waterfront Expansion Analysis", page_icon="🌊", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# TITLE & INTRO
# ──────────────────────────────────────────────────────────────────────────────

st.title("Q2: How could we determine how many more stations to add along the water?")

st.markdown("""
We see the question on whether to expand on the waterfront as a capacity planning question we can 
answer by first comparing **supply to demand on a system-wide level**:

- **Supply**: share of waterfront stations
- **Demand**: share of trip **endpoints** at waterfront stations
- We consider waterfront those stations **within 300m of water**

If demand share > supply share, this is an indication of waterfront being under-served.
Crucially, we are **counting both endpoints (starts + ends) to avoid double-counting 
the waterfront share of "mixed" trips** (happening between waterfront and mainland stations).
""")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Configuration")

wf_input = st.sidebar.text_input(
    "Waterfront Stations CSV",
    value=r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data\waterfront_stations.csv",
    help="Path to waterfront_stations.csv file"
)

hotspot_input = st.sidebar.text_input(
    "Hotspot Hourly Summary CSV",
    value=r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data\waterfront_hotspot_hourly_summary.csv",
    help="Path to waterfront_hotspot_hourly_summary.csv file"
)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(wf_path, hotspot_path):
    """Load pre-computed waterfront analysis data."""
    wf = pd.read_csv(wf_path)
    hotspot_hourly_summary = pd.read_csv(hotspot_path)
    return wf, hotspot_hourly_summary

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

try:
    with st.spinner("Loading pre-computed analysis data…"):
        wf, hotspot_hourly_summary = load_data(wf_input, hotspot_input)
    
    st.success("We are working with pre-computed files. For the analysis from scratch, please see the notebook NYC_Q2_Waterfront_Expansion.ipynb")

    # ──────────────────────────────────────────────────────────────────────────
    # HARDCODED METRICS (from full analysis)
    # ──────────────────────────────────────────────────────────────────────────
    
    total_stations = 1842
    waterfront_stations = 273
    total_waterfront_endpoints = 8452183
    demand_share = 0.142
    supply_share = 0.148
    shortfall = 0  # No gap since supply > demand

    # ── Supply vs Demand
    st.markdown("---")
    st.markdown("We know that at systemic-level we have **No System-Wide Waterfront Shortfall — supply ≈ demand**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Waterfront Stations Share", f"{supply_share:.1%}")
    with c2:
        st.metric("Waterfront Endpoints Share", f"{demand_share:.1%}")
    with c3:
        st.metric("Station Gap", "0")

    st.markdown("""
As this approach uses revealed demand (actual trips taken) as the baseline, it may underestimate true 
demand if potential users were deterred by bike shortages or full docking stations at waterfront locations. 
To better determine whether there is need for expansion we also check:

1. **Concentration of endpoints among waterfront stations (volume concentration)** — thus identifying 
   hotspots among the waterfront stations
2. **Time-concentration among those hotspots** — how busy are these stations during their busiest hours? 
   What is the fraction of total day endpoints that these busy hours generate for hotspot stations?
""")

    # ──────────────────────────────────────────────────────────────────────────
    # CONCENTRATION CHART
    # ──────────────────────────────────────────────────────────────────────────
    
    st.markdown("---")
    st.subheader("Waterfront Station Concentration")

    # Build buckets from dataframe
    wf["cum_share"] = wf["share_of_water_endpoints"].cumsum()
    total_stations_wf = len(wf)
    top10_share = wf["share_of_water_endpoints"].head(10).sum()
    top10pct_n = max(1, int(round(0.10 * total_stations_wf)))
    top10pct_share = wf["share_of_water_endpoints"].head(top10pct_n).sum()
    hotspots_n = int((wf["cum_share"] <= 0.50).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Top 10 Share", f"{top10_share:.1%}")
    c2.metric(f"Top 10% ({top10pct_n}) Share", f"{top10pct_share:.1%}")
    c3.metric("Hotspot Count (50%)", hotspots_n)

    # Segment widths
    seg1 = top10_share
    seg2 = max(0.0, top10pct_share - top10_share)
    seg3 = max(0.0, 0.50 - top10pct_share)
    seg4 = 0.50

    counts = {
        "seg1": 10,
        "seg2": max(0, top10pct_n - 10),
        "seg3": max(0, hotspots_n - top10pct_n),
        "seg4": max(0, total_stations_wf - hotspots_n),
    }

    fig = go.Figure()
    
    fig.add_bar(
        name="Top 10 stations",
        x=[seg1], y=[""],
        orientation="h",
        marker_color="#8B0000",
        text=["Top 10 stations"], textposition="inside", insidetextanchor="middle",
        textfont_color="white",
        hovertemplate="Share: %{x:.1%}<extra></extra>"
    )
    
    fig.add_bar(
        name="Top 10% stations",
        x=[seg2], y=[""],
        orientation="h",
        marker_color="#B22222",
        text=["Top 10% stations"], textposition="inside", insidetextanchor="middle",
        textfont_color="white",
        hovertemplate="Next " + str(counts['seg2']) + " stations<br>Added share: %{x:.1%}<extra></extra>"
    )
    
    fig.add_bar(
        name=f"Top {hotspots_n} stations",
        x=[seg3], y=[""],
        orientation="h",
        marker_color="#DC6E6E",
        text=[f"Top {hotspots_n} stations"], textposition="inside", insidetextanchor="middle",
        textfont_color="white",
        hovertemplate="Next " + str(counts['seg3']) + " stations<br>Added share: %{x:.1%}<extra></extra>"
    )
    
    fig.add_bar(
        name=f"Remaining {counts['seg4']} stations",
        x=[seg4], y=[""],
        orientation="h",
        marker_color="#F4CACA",
        text=[f"Remaining {counts['seg4']} stations"], textposition="inside", insidetextanchor="middle",
        hoverinfo="skip"
    )
    
    fig.update_layout(
        barmode="stack",
        height=220,
        margin=dict(l=30, r=20, t=50, b=40),
        title="Distribution of Endpoints among Waterfront Stations",
        showlegend=False,
        xaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            tickmode="array",
            tickvals=[0, 0.5, 1.0],
            ticktext=["0%", "50%", "100%"],
            title=None
        ),
        yaxis=dict(showticklabels=False, title=None)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TIMING CONCENTRATION & CANDIDATES
    # ──────────────────────────────────────────────────────────────────────────
    
    st.markdown("---")
    st.subheader("Timing Concentration & Expansion Candidates")

    df = hotspot_hourly_summary.copy()
    
    # Candidate rule
    cand_mask = (df["peak_share"] >= 0.12) | ((df["top3_share"] >= 0.30) & (df["span50_hours"] <= 6))
    df["is_candidate"] = np.where(cand_mask, "Candidate", "Not candidate")
    line_w = np.where(df["span50_hours"] <= 6, 2.5, 1.0)

    # Figure
    fig2 = go.Figure()
    
    for group, color in [("Candidate", "#B22222"), ("Not candidate", "#CFCFCF")]:
        sub = df[df["is_candidate"] == group]
        if len(sub) == 0:
            continue
        size = np.clip(sub["total_trips"] / sub["total_trips"].max() * 28, 8, 28)
        fig2.add_scatter(
            x=sub["top3_share"], y=sub["peak_share"], mode="markers",
            marker=dict(size=size, color=color, line=dict(color="#333", width=line_w[sub.index]),
                        opacity=0.95 if group=="Candidate" else 0.75),
            name=group,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Peak hr: %{customdata[1]}:00<br>"
                "Peak share: %{y:.1%}<br>"
                "Top-3 share: %{x:.1%}<br>"
                "span50: %{customdata[2]} hrs<br>"
                "Trips: %{customdata[3]:,}<extra></extra>"
            ),
            customdata=np.stack([sub["station_name"], sub["peak_hour"],
                                 sub["span50_hours"], sub["total_trips"]], axis=1)
        )
    
    # Threshold rules
    fig2.add_hline(y=0.12, line_dash="dash", line_color="#888")
    fig2.add_vline(x=0.30, line_dash="dash", line_color="#888")
    
    fig2.update_layout(
        title="Waterfront Hotspots: Trip Timing and Expansion Candidates",
        xaxis=dict(title="Top-3 hours share", tickformat=".0%", range=[0.22, 0.50]),
        yaxis=dict(title="Peak hour share", tickformat=".0%", range=[0.00, 0.20]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=560, margin=dict(l=60, r=20, t=70, b=60),
        hoverlabel=dict(bgcolor="white")
    )
    
    # Rule note
    fig2.add_annotation(
        xref="paper", yref="paper", x=0.88, y=0.63, showarrow=False,
        text="Flag if peak share ≥ 12% OR (top-3 share ≥ 30% and span50 ≤ 6)",
        font=dict(size=11, color="#555"), align="left"
    )
    
    # Legend box
    fig2.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0.46, x1=0.995, y0=0.04, y1=0.26,
        line=dict(color="rgba(50,50,50,0.30)"),
        fillcolor="rgba(245,245,245,0.92)",
        layer="below"
    )
    
    fig2.add_annotation(
        xref="paper", yref="paper", x=0.465, y=0.255,
        xanchor="left", yanchor="top", showarrow=False, align="left",
        text=(
            "<b>Legend</b><br>" +
            "<b>Peak hour</b> — single busiest hour of the day for a station<br>" +
            "<b>Peak share</b> — that peak hour's share of the station's daily trips<br>" +
            "<b>Top-3 share</b> — combined share of the three busiest hours<br>" +
            "<i>(if demand were even across ~17 \"active\" hours, three ≈ 17.6%)</i><br>" +
            "<b>span50</b> — fewest hours (starting from busiest) to reach 50% of trips"
        ),
        font=dict(size=11, color="#333")
    )
    
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
We flag as candidates for expansion those hotspot stations where the **single busiest hour accounts for 12%**+ of daily trips, **or** the **top three hours combined exceed 30%**, **and** the **number of hours needed to reach 50% of trips is under 6**.

Using **AND** prevents false positives—‘busy-but-spread’ stations that show high activity across many hours. Those are better addressed through operational measures, not new infrastructure
""")

    # ──────────────────────────────────────────────────────────────────────────
    # RECOMMENDATIONS
    # ──────────────────────────────────────────────────────────────────────────
    
    st.markdown("---")
    st.subheader("Recommendations")

    st.markdown("""
With a 300 m definition and station-count supply, no additional waterfront stations are needed system-wide 
(demand ≈ supply). While 50% of trip endpoints happen in just 40 of the 273 waterfront stations, **only few stations show remarkable intensity in both volume and time-concentration**. 
We **recommend thus against blanket expansion**, but rather starting **expansion around these 7 identified stations**:

1. North Moore St & Greenwich St
2. Vesey Pl & River Terrace
3. West St & Chambers St
4. W 15 St & 10 Ave
5. 1 Ave & E 44 St
6. 5 Ave & E 87 St
7. 10 Ave & W 14 St
""")

    st.download_button(
        "Download Waterfront Station Data (CSV)",
        data=wf.to_csv(index=False),
        file_name="waterfront_stations_analysis.csv",
        mime="text/csv",
    )

except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.info("Update the file paths in the left sidebar.")
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please check your data files and try again.")