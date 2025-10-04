# Data files required for this dashboard are hosted on Google Drive:
# https://drive.google.com/drive/folders/18dn7QjYPa3z1ZIkUEGr9zHzWnwQxSUMz

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter
import warnings
import osmnx as ox
import geopandas as gpd

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Waterfront Expansion Analysis", page_icon="🌊", layout="wide")

# =============================================================================
# TITLE AND INTRODUCTION
# =============================================================================

st.title("Q2: How could we determine how many more stations to add along the water?")

st.markdown("""
We see the question on whether to expand on the waterfront as capacity planning question that we can answer by comparing **supply to demand**:

- **Supply**: The share of stations located within 300m of water (waterfront buffer)
- **Demand**: The share of trip endpoints (starts OR ends) that occur at waterfront stations

If waterfront trips account for a greater proportion of system usage than waterfront stations 
represent of total capacity, this indicates the waterfront is under-served relative to demand.

**Note**: We measure the fraction of **all trip endpoints** (not trips) that occur at 
waterfront stations, creating an apples-to-apples comparison with the fraction of stations 
that are waterfront.
""")

# =============================================================================
# CONFIGURATION & DATA PATHS
# =============================================================================

st.sidebar.header("Configuration")

default_trips_path = r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data\nyc_2022_essential_data.csv"
default_station_path = r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data\station_to_nta.csv"

trips_path = st.sidebar.text_input(
    "Trips data path:",
    value=default_trips_path,
    help="Path to NYC CitiBike 2022 trips data"
)

station_path = st.sidebar.text_input(
    "Station data path:",
    value=default_station_path,
    help="Path to station_to_nta.csv file"
)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def get_waterfront_buffer():
    """Get the 300m land-side waterfront buffer using OSM data."""
    METRIC = "EPSG:26918"  # meters (UTM 18N)
    
    # NYC polygon
    nyc_4326 = ox.geocode_to_gdf("New York City, New York, USA").to_crs(4326)
    poly_4326 = nyc_4326.geometry.iloc[0]
    nyc_m = nyc_4326.to_crs(METRIC)
    nyc_poly_m = nyc_m.geometry.iloc[0]
    
    def safe_features_m(poly_4326, tags):
        try:
            g = ox.features_from_polygon(poly_4326, tags)
            if g.empty:
                return gpd.GeoDataFrame(geometry=[], crs=METRIC)
            return g.to_crs(METRIC)
        except Exception:
            return gpd.GeoDataFrame(geometry=[], crs=METRIC)
    
    # Fetch water polygons + coastline
    water_poly_m = safe_features_m(poly_4326, {"natural": "water"})
    coastline_m = safe_features_m(poly_4326, {"natural": "coastline"})
    
    # Keep big/tidal features; drop tiny ponds
    keep_types = {"river", "sea", "bay", "harbour", "estuary", "tidal_channel"}
    if not water_poly_m.empty:
        water_poly_m["area_m2"] = water_poly_m.geometry.area
        wtype = water_poly_m.get("water").fillna("")
        mask_type = wtype.isin(keep_types)
        mask_area = water_poly_m["area_m2"] >= 1e5  # ≥ 0.1 km²
        water_big = water_poly_m[mask_type | mask_area][["geometry"]].copy()
    else:
        water_big = gpd.GeoDataFrame(geometry=[], crs=METRIC)
    
    layers = []
    if not water_big.empty:
        layers.append(water_big)
    if not coastline_m.empty:
        coast = coastline_m.copy()
        coast["geometry"] = coast.buffer(10)  # 10 m ribbon
        layers.append(coast[["geometry"]])
    
    if not layers:
        raise RuntimeError("No suitable water features returned from OSM.")
    
    water_m_filt = pd.concat(layers, ignore_index=True)
    water_union = water_m_filt.union_all()
    
    # Land-side 300 m ribbon
    land = nyc_poly_m.difference(water_union.buffer(1))
    ribbon_land = water_union.buffer(300).intersection(land)
    
    return ribbon_land, METRIC

@st.cache_data
def load_station_data(file_path):
    """Load and process station data with waterfront flags using OSM buffer."""
    station_to_nta = pd.read_csv(file_path)
    
    # Normalize and aggregate stations
    s = station_to_nta.copy()
    s["station_key"] = s["station_name"].str.strip().str.lower()
    
    def _mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else x.iloc[0]
    
    unique = (s.groupby("station_key", as_index=False)
               .agg(station_name=("station_name", "first"),
                    lat=("lat", "median"),
                    lng=("lng", "median"),
                    borough=("borough", _mode)))
    
    # Get waterfront buffer
    ribbon_land, METRIC = get_waterfront_buffer()
    
    # Convert to GeoDataFrame in metric CRS
    stations_gdf = gpd.GeoDataFrame(
        unique,
        geometry=gpd.points_from_xy(unique["lng"], unique["lat"]),
        crs="EPSG:4326"
    ).to_crs(METRIC)
    
    # Create a GeoDataFrame for the buffer polygon
    buffer_gdf = gpd.GeoDataFrame({'geometry': [ribbon_land]}, crs=METRIC)
    
    # Use spatial join to find stations within buffer
    stations_in_buffer = gpd.sjoin(stations_gdf, buffer_gdf, how='left', predicate='within')
    stations_gdf["near_water"] = stations_in_buffer.index_right.notna()
    
    # Return as regular DataFrame
    return stations_gdf[["station_key", "station_name", "lat", "lng", "borough", "near_water"]].copy()

@st.cache_data
def calculate_endpoint_shares(trips_path, station_flags):
    """Calculate supply and demand shares."""
    flag_map = dict(zip(station_flags['station_key'], station_flags['near_water']))
    
    supply_share = station_flags['near_water'].mean()
    total_stations = len(station_flags)
    
    endpoints_water = 0
    total_trips = 0
    
    CHUNK = 500_000
    usecols = ["start_station_name", "end_station_name"]
    dtype = {"start_station_name": "string", "end_station_name": "string"}
    
    for chunk in pd.read_csv(trips_path, usecols=usecols, dtype=dtype, 
                             chunksize=CHUNK, low_memory=False):
        s = chunk["start_station_name"].str.strip().str.lower().map(flag_map).fillna(False)
        e = chunk["end_station_name"].str.strip().str.lower().map(flag_map).fillna(False)
        endpoints_water += int(s.sum() + e.sum())
        total_trips += len(chunk)
    
    demand_share = endpoints_water / (2 * total_trips) if total_trips else 0.0
    shortfall = max(0, int(round((demand_share - supply_share) * total_stations)))
    
    return {
        'supply_share': supply_share,
        'demand_share': demand_share,
        'total_stations': total_stations,
        'waterfront_stations': int(station_flags['near_water'].sum()),
        'shortfall': shortfall,
        'total_trips': total_trips,
        'waterfront_endpoints': endpoints_water
    }

@st.cache_data
def calculate_station_activity(trips_path, station_flags):
    """Calculate trip counts per waterfront station."""
    water_keys = set(station_flags[station_flags['near_water']]['station_key'])
    
    counts = Counter()
    CHUNK = 500_000
    usecols = ["start_station_name", "end_station_name"]
    dtype = {"start_station_name": "string", "end_station_name": "string"}
    
    for chunk in pd.read_csv(trips_path, usecols=usecols, dtype=dtype,
                             chunksize=CHUNK, low_memory=False):
        s = chunk["start_station_name"].str.strip().str.lower()
        e = chunk["end_station_name"].str.strip().str.lower()
        counts.update(s[s.isin(water_keys)].value_counts().to_dict())
        counts.update(e[e.isin(water_keys)].value_counts().to_dict())
    
    wf_counts = pd.DataFrame({
        "station_key": list(counts.keys()),
        "endpoints": list(counts.values())
    })
    
    wf_counts = wf_counts.merge(
        station_flags[['station_key', 'station_name', 'borough']],
        on='station_key',
        how='left'
    )
    
    wf_counts = wf_counts.sort_values('endpoints', ascending=False).reset_index(drop=True)
    wf_counts['share_of_water_endpoints'] = wf_counts['endpoints'] / wf_counts['endpoints'].sum()
    wf_counts['cum_share'] = wf_counts['share_of_water_endpoints'].cumsum()
    
    return wf_counts

@st.cache_data
def calculate_hourly_patterns(trips_path, station_flags, wf_counts):
    """Calculate hourly demand patterns for TOP waterfront hotspots only (those accounting for 50% of endpoints)."""
    from collections import defaultdict
    
    # Filter to only hotspot stations (top 50% of waterfront endpoints)
    hotspots = wf_counts[wf_counts['cum_share'] <= 0.50].copy()
    hotspots['station_key'] = hotspots['station_name'].str.strip().str.lower()
    hotspot_keys = set(hotspots['station_key'])
    key_to_name = dict(zip(hotspots['station_key'], hotspots['station_name']))
    
    # 24-hour counters
    hour_counts = {k: np.zeros(24, dtype=np.int64) for k in hotspot_keys}
    
    CHUNK = 500_000
    usecols = ["start_station_name", "end_station_name", "started_at", "ended_at"]
    dtype = {"start_station_name": "string", "end_station_name": "string"}
    
    for chunk in pd.read_csv(trips_path, usecols=usecols, dtype=dtype, chunksize=CHUNK, low_memory=False):
        s_key = chunk["start_station_name"].str.strip().str.lower()
        e_key = chunk["end_station_name"].str.strip().str.lower()
        mask_hot = s_key.isin(hotspot_keys) | e_key.isin(hotspot_keys)
        
        if not mask_hot.any():
            continue
        
        sub = chunk.loc[mask_hot]
        s_key = s_key[mask_hot]
        e_key = e_key[mask_hot]
        s_hr = pd.to_datetime(sub["started_at"], errors="coerce").dt.hour
        e_hr = pd.to_datetime(sub["ended_at"], errors="coerce").dt.hour
        
        involved = pd.Index(s_key[s_key.isin(hotspot_keys)].unique()).union(
                   pd.Index(e_key[e_key.isin(hotspot_keys)].unique()))
        
        for k in involved:
            ms = (s_key == k)
            me = (e_key == k) & (~ms)
            
            if ms.any():
                hour_counts[k] += np.bincount(s_hr[ms].dropna().astype(int), minlength=24)
            if me.any():
                hour_counts[k] += np.bincount(e_hr[me].dropna().astype(int), minlength=24)
    
    # Calculate metrics
    rows = []
    for k, counts in hour_counts.items():
        total = counts.sum()
        if total == 0:
            continue
        
        p = counts / total
        peak_hour = int(np.argmax(counts))
        peak_share = float(counts.max() / total)
        top3_share = float(np.sort(p)[-3:].sum())
        span50 = int((np.sort(p)[::-1].cumsum() <= 0.5).sum() + 1)
        
        rows.append({
            "station_name": key_to_name.get(k, k),
            "total_trips": int(total),
            "peak_hour": peak_hour,
            "peak_share": round(peak_share, 3),
            "top3_share": round(top3_share, 3),
            "span50_hours": span50
        })
    
    return pd.DataFrame(rows)

# =============================================================================
# MAIN APP
# =============================================================================

try:
    with st.spinner("Loading geographic data and identifying waterfront stations... This may take a moment..."):
        stations = load_station_data(station_path)
    
    waterfront_count = int(stations['near_water'].sum())
    st.success(f"✓ Loaded {len(stations)} unique stations ({waterfront_count} waterfront, {len(stations) - waterfront_count} inland)")
    
    with st.expander("Methodology"):
        st.markdown("""
        **Analytical Approach:**
        
        1. **Define waterfront zone**: 300m buffer from water bodies (rivers, harbors, coastline)
        2. **Calculate supply**: % of stations within waterfront buffer
        3. **Calculate demand**: % of trip endpoints at waterfront stations
        4. **Compare**: Gap between demand and supply indicates capacity shortfall
        
        **Key Refinement:**
        
        We initially tried counting trips that "touch" waterfront (start OR end), but this 
        double-counted mixed trips (one end waterfront, one end inland). So we count individual endpoints to 
        create direct apples-to-apples comparison with station share.
        """)
    
    # Calculate metrics
    with st.spinner("Analyzing trip patterns... This may take a minute..."):
        metrics = calculate_endpoint_shares(trips_path, stations)
        wf_counts = calculate_station_activity(trips_path, stations)
        hourly_data = calculate_hourly_patterns(trips_path, stations, wf_counts)
    
    st.markdown("---")
    st.subheader("Supply vs. Demand Analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Stations",
            f"{metrics['total_stations']:,}",
            help="All unique stations in system"
        )
    
    with col2:
        st.metric(
            "Waterfront Stations",
            f"{metrics['waterfront_stations']}",
            delta=f"{metrics['supply_share']:.1%}",
            help="Stations within 300m of water"
        )
    
    with col3:
        st.metric(
            "Waterfront Endpoints",
            f"{metrics['waterfront_endpoints']:,}",
            delta=f"{metrics['demand_share']:.1%}",
            help="Trip endpoints at waterfront stations"
        )
    
    with col4:
        shortfall_label = "Station Gap" if metrics['shortfall'] > 0 else "No Gap"
        st.metric(
            shortfall_label,
            f"{metrics['shortfall']}",
            delta=f"{(metrics['demand_share'] - metrics['supply_share']):.1%}",
            delta_color="inverse" if metrics['shortfall'] > 0 else "off",
            help="Implied station shortfall"
        )
    
    # Conclusion box
    if metrics['shortfall'] == 0:
        st.success(f"""
        **✅ No System-Wide Waterfront Shortfall**
        
        Endpoint share ({metrics['demand_share']:.1%}) ≈ Station share ({metrics['supply_share']:.1%}) → No system-wide capacity gap.
        
        However, **concentration analysis** below identifies specific high-demand hotspots 
        that may benefit from targeted expansion.
        """)
    else:
        st.warning(f"""
        **⚠️ Potential Waterfront Capacity Gap**
        
        Demand share ({metrics['demand_share']:.1%}) > Supply share ({metrics['supply_share']:.1%})
        
        Implied shortfall: **~{metrics['shortfall']} stations**
        
        See concentration analysis below for specific locations.
        """)
    
    # =============================================================================
    # CONCENTRATION ANALYSIS
    # =============================================================================
    
    st.markdown("---")
    st.subheader("Waterfront Station Concentration")
    
    st.markdown("""
    Even without system-wide shortfall, we notice that demand is highly concentrated at a few hotspots. 
    Understanding this concentration helps determine the need for and localize any expansion efforts.
    """)
    
    # Calculate concentration metrics
    top10_share = wf_counts['share_of_water_endpoints'].head(10).sum()
    top10pct_n = max(1, int(round(0.10 * len(wf_counts))))
    top10pct_share = wf_counts['share_of_water_endpoints'].head(top10pct_n).sum()
    hotspots_n = int((wf_counts['cum_share'] <= 0.50).sum())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Top 10 Stations Share",
            f"{top10_share:.1%}",
            help="% of waterfront endpoints at busiest 10 stations"
        )
    
    with col2:
        st.metric(
            f"Top 10% ({top10pct_n}) Share",
            f"{top10pct_share:.1%}",
            help="% of waterfront endpoints at top 10% of stations"
        )
    
    with col3:
        st.metric(
            "Hotspot Count (50%)",
            hotspots_n,
            help="Number of stations accounting for 50% of waterfront endpoints"
        )
    
    # Concentration visualization
    seg1 = top10_share
    seg2 = max(0.0, top10pct_share - top10_share)
    seg3 = max(0.0, 0.50 - top10pct_share)
    seg4 = 0.50
    
    counts = {
        "seg2": max(0, top10pct_n - 10),
        "seg3": max(0, hotspots_n - top10pct_n),
        "seg4": max(0, len(wf_counts) - hotspots_n),
    }
    
    fig = go.Figure()
    
    fig.add_bar(
        name="Top 10",
        x=[seg1], y=[""],
        orientation="h",
        marker_color="#8B0000",
        text=["Top 10 stations"],
        textposition="inside",
        insidetextanchor="middle",
        textfont_color="white",
        hovertemplate="Share: %{x:.1%}<extra></extra>"
    )
    
    fig.add_bar(
        name="Top 10%",
        x=[seg2], y=[""],
        orientation="h",
        marker_color="#B22222",
        text=["Top 10% stations"],
        textposition="inside",
        insidetextanchor="middle",
        textfont_color="white",
        hovertemplate=f"Next {counts['seg2']} stations<br>Added share: %{{x:.1%}}<extra></extra>"
    )
    
    fig.add_bar(
        name=f"Top {hotspots_n}",
        x=[seg3], y=[""],
        orientation="h",
        marker_color="#DC6E6E",
        text=[f"Top {hotspots_n} stations"],
        textposition="inside",
        insidetextanchor="middle",
        textfont_color="white",
        hovertemplate=f"Next {counts['seg3']} stations<br>Added share: %{{x:.1%}}<extra></extra>"
    )
    
    fig.add_bar(
        name="Remaining",
        x=[seg4], y=[""],
        orientation="h",
        marker_color="#F4CACA",
        text=[f"Remaining {counts['seg4']} stations"],
        textposition="inside",
        hoverinfo="skip"
    )
    
    fig.update_layout(
        barmode="stack",
        height=200,
        margin=dict(l=30, r=20, t=40, b=40),
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
    
    # =============================================================================
    # TIMING CONCENTRATION & EXPANSION CANDIDATES
    # =============================================================================
    
    st.markdown("---")
    st.subheader("Timing Concentration & Expansion Candidates")
    
    st.markdown("""
    Concentration alone doesn't prove we need new stations. If demand at those stations is spread 
    throughout the day, operations can usually manage without additional infrastructure. However, 
    **if a large share of demand clusters in a few hours** (e.g., peak hour ≥ 12% or top-3 hours 
    ≥ 30% with span50 ≤ 6), **adding small satellite stations nearby can help absorb latent demand 
    previously constrained by full/empty docks**.
    
    **Our candidate rule:** Flag stations if **peak_share ≥ 12%** OR (**top3_share ≥ 30%** AND **span50 ≤ 6**).
    
    The logic: A 2-3 hour demand wave is meaningful only if the whole day is tight (half the day's 
    trips concentrated in ≤6 hours). Using AND prevents false positives from "busy-but-spread" 
    stations that have moderately high hours but operate across many hours—situations better 
    handled through operational adjustments rather than new infrastructure.
    """)
    
    # Create candidate visualization
    df = hourly_data.copy()
    cand_mask = (df["peak_share"] >= 0.12) | ((df["top3_share"] >= 0.30) & (df["span50_hours"] <= 6))
    df["is_candidate"] = np.where(cand_mask, "Candidate", "Not candidate")
    line_w = np.where(df["span50_hours"] <= 6, 2.5, 1.0)
    
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
    
    fig2.add_annotation(
        xref="paper", yref="paper", x=0.88, y=0.63, showarrow=False,
        text="Flag if peak share ≥ 12% OR (top-3 share ≥ 30% and span50 ≤ 6)",
        font=dict(size=11, color="#555"), align="left"
    )
    
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
            "<b>Peak hour</b> – single busiest hour of the day for a station<br>" +
            "<b>Peak share</b> – that peak hour's share of the station's daily trips<br>" +
            "<b>Top-3 share</b> – combined share of the three busiest hours<br>" +
            "<i>(if demand were even across ~17 \"active\" hours, three ≈ 17.6%)</i><br>" +
            "<b>span50</b> – fewest hours (starting from busiest) to reach 50% of trips"
        ),
        font=dict(size=11, color="#333")
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # =============================================================================
    # RECOMMENDATIONS
    # =============================================================================
    
    st.markdown("---")
    st.subheader("Recommendations")
    
    candidate_count = int(cand_mask.sum())
    
    st.markdown(f"""
    **With 300m definition and station-count supply:**
    
    - **System balance**: Endpoint share ({metrics['demand_share']:.1%}) ≈ Station share ({metrics['supply_share']:.1%})
    - **Concentration**: {hotspots_n} stations account for 50% of waterfront endpoints
    - **Hotspot intensity**: Top 10 stations = {top10_share:.1%} of all waterfront activity
    - **Expansion candidates**: {candidate_count} stations show concentrated peak-hour demand
    
    **Strategic Approach:**
    
    1. **No blanket expansion needed** - Overall supply matches demand
    
    2. **Targeted pilot recommendation** - Focus on the {candidate_count} stations showing:
       - Highest activity concentration
       - Peak-hour congestion (>12% of daily trips in single hour OR >30% in top-3 hours with tight daily spread)
       
    3. **Implementation**: Add small satellite stations (10-15 docks) within 100-200m of flagged hotspots 
       to absorb overflow during peak hours without disrupting existing operations
    """)
    
    # Download option
    st.markdown("---")
    csv = wf_counts.to_csv(index=False)
    st.download_button(
        label="Download Waterfront Station Data (CSV)",
        data=csv,
        file_name="waterfront_stations_analysis.csv",
        mime="text/csv"
    )

except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}")
    st.info("Please update the file paths in the sidebar.")
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please check your data files and try again.")