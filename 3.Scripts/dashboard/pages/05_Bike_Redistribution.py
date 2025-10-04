import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import MiniBatchKMeans
from shapely.geometry import MultiPoint
import warnings
import math

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Q3: How to approach Station Restocking in NYC?", page_icon="🔄", layout="wide")

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH = r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data\nyc_2022_essential_data.csv"
TOP_N = 300
TARGET_STATIONS_PER_ZONE = 12
SMOOTH_WINDOW = 3
EPS = 1.0
MIN_BLOCK_HOURS = 2
MIN_ABS_C = 6
CONSOLIDATION_TOLERANCE = 3
OVERNIGHT_START = 23
OVERNIGHT_END = 4
RESIDENTIAL_MORNING_THRESHOLD = -20
RESIDENTIAL_EVENING_THRESHOLD = 20
RESIDENTIAL_MIDDAY_THRESHOLD = 15
FLIP_PENALTY = 0.5

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def greedy_matching(supply_zones, demand_zones, top_k=15):
    """Greedy nearest-neighbor matching between supply and demand zones."""
    if supply_zones.empty or demand_zones.empty:
        return pd.DataFrame(columns=['supply_zone', 'demand_zone', 'bikes_moved'])
    
    supply = supply_zones.copy()
    demand = demand_zones.copy()
    routes = []
    
    while not supply.empty and not demand.empty and supply['qty'].sum() > 0 and demand['qty'].sum() > 0:
        min_dist = float('inf')
        best_s_idx = None
        best_d_idx = None
        
        for s_idx, s_row in supply.iterrows():
            if s_row['qty'] <= 0:
                continue
            for d_idx, d_row in demand.iterrows():
                if d_row['qty'] <= 0:
                    continue
                dist = haversine_distance(s_row['lat'], s_row['lon'], 
                                         d_row['lat'], d_row['lon'])
                if dist < min_dist:
                    min_dist = dist
                    best_s_idx = s_idx
                    best_d_idx = d_idx
        
        if best_s_idx is None or best_d_idx is None:
            break
        
        bikes_to_move = min(supply.loc[best_s_idx, 'qty'], 
                           demand.loc[best_d_idx, 'qty'])
        
        routes.append({
            'supply_zone': supply.loc[best_s_idx, 'zone'],
            'demand_zone': demand.loc[best_d_idx, 'zone'],
            'bikes_moved': bikes_to_move
        })
        
        supply.loc[best_s_idx, 'qty'] -= bikes_to_move
        demand.loc[best_d_idx, 'qty'] -= bikes_to_move
        
        if len(routes) >= top_k:
            break
    
    return pd.DataFrame(routes).sort_values('bikes_moved', ascending=False) if routes else pd.DataFrame(columns=['supply_zone', 'demand_zone', 'bikes_moved'])

def format_bikes(n):
    """Format bike count for display."""
    return f"{n/1000:.1f}k" if n >= 1000 else str(int(round(n)))

def create_zone_hulls_geojson(stations_df, zones_to_show):
    """Create convex hulls for zones."""
    features = []
    for zone in zones_to_show:
        zone_stations = stations_df[stations_df['geo_zone'] == zone]
        if len(zone_stations) >= 3:
            points = [(row['lon'], row['lat']) for _, row in zone_stations.iterrows()]
            mp = MultiPoint(points)
            hull = mp.convex_hull
            features.append({
                'type': 'Feature',
                'properties': {'zone': int(zone)},
                'geometry': hull.__geo_interface__
            })
    return {'type': 'FeatureCollection', 'features': features}

# =============================================================================
# DATA LOADING AND PROCESSING (CACHED)
# =============================================================================

@st.cache_data
def load_and_process_data(file_path):
    """Load data and perform all preprocessing steps."""
    
    # Load data
    df = pd.read_csv(file_path)
    for col in ["started_at", "ended_at", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Get station coordinates
    starts_geo = (df.dropna(subset=["start_station_name", "start_lat", "start_lng"])
                  .groupby("start_station_name")[["start_lat", "start_lng"]]
                  .median()
                  .rename(columns={"start_lat": "lat", "start_lng": "lon"}))
    
    ends_geo = (df.dropna(subset=["end_station_name", "end_lat", "end_lng"])
                .groupby("end_station_name")[["end_lat", "end_lng"]]
                .median()
                .rename(columns={"end_lat": "lat", "end_lng": "lon"}))
    
    stations_all = starts_geo.combine_first(ends_geo)
    stations_all.index.name = "station"
    
    # Calculate station volumes
    starts_count = df.groupby("start_station_name").size().rename("starts")
    ends_count = df.groupby("end_station_name").size().rename("ends")
    
    station_volume = (pd.concat([starts_count, ends_count], axis=1)
                      .fillna(0)
                      .assign(total=lambda x: x.starts + x.ends)
                      .sort_values("total", ascending=False))
    
    # Keep top stations
    top_stations = (station_volume.head(TOP_N)
                    .join(stations_all, how="left")
                    .dropna(subset=["lat", "lon"])
                    .copy())
    
    # Cluster into zones
    n_clusters = max(1, math.ceil(len(top_stations) / TARGET_STATIONS_PER_ZONE))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, 
                             batch_size=256, n_init="auto")
    
    top_stations["geo_zone"] = kmeans.fit_predict(top_stations[["lat", "lon"]])
    zone_mapping = {old: new for new, old in enumerate(sorted(top_stations["geo_zone"].unique()))}
    top_stations["geo_zone"] = top_stations["geo_zone"].map(zone_mapping).astype(int)
    
    station_to_zone = top_stations["geo_zone"].to_dict()
    
    # Calculate hourly flows
    starts_hourly = (df.loc[df["start_station_name"].isin(top_stations.index)]
                     .dropna(subset=["started_at"])
                     .assign(zone=lambda x: x["start_station_name"].map(station_to_zone),
                            hour=lambda x: x["started_at"].dt.hour,
                            dow=lambda x: x["started_at"].dt.dayofweek,
                            month=lambda x: x["started_at"].dt.to_period("M").astype(str))
                     .groupby(["month", "dow", "hour", "zone"])
                     .size()
                     .rename("starts")
                     .reset_index())
    
    ends_hourly = (df.loc[df["end_station_name"].isin(top_stations.index)]
                   .dropna(subset=["ended_at"])
                   .assign(zone=lambda x: x["end_station_name"].map(station_to_zone),
                          hour=lambda x: x["ended_at"].dt.hour,
                          dow=lambda x: x["ended_at"].dt.dayofweek,
                          month=lambda x: x["ended_at"].dt.to_period("M").astype(str))
                   .groupby(["month", "dow", "hour", "zone"])
                   .size()
                   .rename("ends")
                   .reset_index())
    
    # Create baseline
    all_months = sorted(set(starts_hourly["month"]) | set(ends_hourly["month"]))
    all_dows = sorted(set(starts_hourly["dow"]) | set(ends_hourly["dow"]))
    all_hours = range(24)
    all_zones = sorted(top_stations["geo_zone"].unique())
    
    grid = pd.MultiIndex.from_product(
        [all_months, all_dows, all_hours, all_zones],
        names=["month", "dow", "hour", "zone"]
    ).to_frame(index=False)
    
    baseline_hourly = (grid.merge(starts_hourly, on=["month", "dow", "hour", "zone"], how="left")
                       .merge(ends_hourly, on=["month", "dow", "hour", "zone"], how="left")
                       .fillna({"starts": 0, "ends": 0}))
    
    baseline_hourly["C"] = baseline_hourly["ends"] - baseline_hourly["starts"]
    baseline_hourly["C_smooth"] = (baseline_hourly
        .sort_values(["month", "dow", "zone", "hour"])
        .groupby(["month", "dow", "zone"])["C"]
        .transform(lambda x: x.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()))
    
    stations_top = top_stations.rename_axis("station").reset_index()[["station", "lat", "lon", "geo_zone"]]
    
    return {
        'stations_top': stations_top,
        'baseline_hourly': baseline_hourly,
        'all_months': all_months,
        'all_dows': all_dows,
        'all_zones': all_zones
    }

# =============================================================================
# STREAMLIT APP
# =============================================================================

st.title("Q3: How to approach Station Restocking in NYC?")

st.markdown("""
This model uses **zone-based rebalancing** to identify when and where bike redistribution 
is needed for a given month × day of week (dow) and relevant time blocks for that month×dow, 
reflecting the variability of bike usage at various time scales. The system:

1. **Groups 300 top Manhattan stations into 25 geographic zones**
2. **Detects time blocks** when zones act as supply (excess bikes) or demand (need bikes) - 
   *Note: For dashboard simplicity, this version uses predefined time periods (Morning Commute, 
   Midday, etc.). The full model detects granular time blocks specific to each month×day 
   combination (e.g., "04:00–13:00", "14:00–19:00") with ±3 hour consolidation tolerance.*
3. **Excludes self-correcting residential zones** that balance naturally
4. **Adjusts for bidirectional demand** to avoid stripping busy stations bare
5. **Suggests redistribution routes** using greedy nearest-neighbor matching

**Note:** This is a simplified interactive version. The full model includes block consolidation,
sticky zones detection, and other features described in the methodology.
""")

# Sidebar for file path
st.sidebar.header("Configuration")
file_path = st.sidebar.text_input(
    "Data file path:",
    value=CSV_PATH,
    help="Path to NYC CitiBike 2022 data"
)

# Load data
try:
    with st.spinner("Loading and processing data... This may take a minute..."):
        data = load_and_process_data(file_path)
    
    st.success(f"✓ Processed {len(data['stations_top'])} stations in {len(data['all_zones'])} zones")
    
    # Display methodology
    with st.expander("Model Methodology"):
        st.markdown("""
        **Solution Approach:**
        
        1. **Geographic zones**: K-means clustering on coordinates (25 zones)
        2. **Block detection**: Identify continuous supply/demand periods
        3. **Block consolidation**: Merge similar time windows (4,644 → 731 blocks)
        4. **Self-correcting zones**: Exclude residential areas with natural daily rhythms
        5. **Bidirectional adjustment**: Scale moves by rental/return pressure (53% of raw)
        6. **Sticky zones**: Reduce moves for zones that flip roles (31.7% of blocks)
        7. **Greedy matching**: Connect supply to demand via nearest-neighbor routes
        8. **User controls**: β parameter scales intervention intensity
        
        **Implementation Note for This Dashboard:**
        
        For performance and usability, this interactive version uses simplified time blocks (Morning Commute, 
        Midday, Evening Commute, Night, Overnight) rather than the granular, dynamically-detected blocks from 
        the full model. The complete implementation with month×dow-specific time blocks (731 unique windows 
        after consolidation), overnight canonical blocks, and all filtering steps is available in the Jupyter 
        notebook `NYC_Q3_Zone-Based_Restocking.ipynb`.
        
        **Key Limitations:**
        
        - **Data limitations**: Our model is built only on CitiBike's historical trip 2022 data. It lacks 
          real-time trip updates, as well as traffic, terrain and precipitation data. Therefore, it treats 
          2022's historical data as fully predictive for a day's operational recommendations.
        - **Temporal ambiguity**: Block consolidation (±3 hour tolerance) reduces fragmentation but forces zones 
          with different timing onto shared labels, obscuring precise intervention timing. Additionally, consolidated 
          blocks sometimes overlap (e.g., 06:00-08:00 and 05:00-09:00 coexisting), creating dropdown redundancy. 
          This is partially mitigated by focusing on top 300 Manhattan stations with synchronized demand patterns.
        - **Arbitrary thresholds lack principled derivation**
        - **Suboptimal greedy matching was used out of lack of linear programming skills**
        
        We proceeded with these limitations as a way of engaging with the question's complexity.
        """)
    
    # Interactive controls
    st.markdown("---")
    st.subheader("Interactive Rebalancing Map")
    
    col1, col2, col3 = st.columns(3)
    
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    with col1:
        selected_month = st.selectbox(
            "Month",
            options=data['all_months'],
            format_func=lambda x: pd.Period(x).strftime('%B %Y')
        )
    
    with col2:
        selected_dow = st.selectbox(
            "Day of Week",
            options=data['all_dows'],
            format_func=lambda x: dow_labels[x]
        )
    
    with col3:
        # Simplified time block selection
        time_blocks = [
            ("Morning Commute", 5, 9),
            ("Midday", 10, 14),
            ("Evening Commute", 15, 19),
            ("Night", 20, 23),
            ("Overnight", 23, 4)
        ]
        selected_block_label = st.selectbox(
            "Time Block",
            options=[label for label, _, _ in time_blocks]
        )
        hour_from = [hf for label, hf, _ in time_blocks if label == selected_block_label][0]
        hour_to = [ht for label, _, ht in time_blocks if label == selected_block_label][0]
    
    st.markdown("**Rebalancing Parameters:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        beta = st.slider(
            "β (Intervention Intensity)",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Fraction of movable bikes to actually move (0.1=conservative, 0.9=aggressive)"
        )
    
    with col2:
        min_bikes = st.slider(
            "Minimum Bikes Threshold",
            min_value=0,
            max_value=50,
            value=10,
            step=5,
            help="Hide zones with recommendations below this threshold"
        )
    
    with col3:
        max_routes = st.slider(
            "Maximum Routes Shown",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            help="Limit number of supply→demand routes displayed"
        )
    
    # Filter data for selected time window
    window_data = data['baseline_hourly'][
        (data['baseline_hourly']['month'] == selected_month) &
        (data['baseline_hourly']['dow'] == selected_dow) &
        (data['baseline_hourly']['hour'] >= hour_from) &
        (data['baseline_hourly']['hour'] < hour_to)
    ]
    
    if window_data.empty:
        st.warning("No data available for this selection")
    else:
        # Aggregate by zone
        zone_summary = window_data.groupby('zone').agg({
            'C': 'sum',
            'starts': 'sum',
            'ends': 'sum'
        }).reset_index()
        
        zone_summary['total_activity'] = zone_summary['starts'] + zone_summary['ends']
        zone_summary['rental_pressure'] = np.where(
            zone_summary['total_activity'] > 0,
            zone_summary['starts'] / zone_summary['total_activity'],
            0.5
        )
        zone_summary['return_pressure'] = np.where(
            zone_summary['total_activity'] > 0,
            zone_summary['ends'] / zone_summary['total_activity'],
            0.5
        )
        
        # Simplified movable calculation
        zone_summary['action'] = np.where(zone_summary['C'] > 0, 'supply', 'demand')
        zone_summary['movable'] = np.where(
            zone_summary['action'] == 'supply',
            abs(zone_summary['C']) * zone_summary['return_pressure'],
            abs(zone_summary['C']) * zone_summary['rental_pressure']
        )
        
        # Apply beta and filter
        zone_summary['qty'] = (zone_summary['movable'] * beta).round()
        zone_summary = zone_summary[
            (zone_summary['qty'] >= min_bikes) &
            (abs(zone_summary['C']) >= MIN_ABS_C)
        ]
        
        # Get centroids
        centroids = data['stations_top'].groupby('geo_zone')[['lat', 'lon']].mean().reset_index().rename(columns={'geo_zone': 'zone'})
        zone_data = zone_summary.merge(centroids, on='zone', how='left')
        
        supply = zone_data[zone_data['action'] == 'supply'][['zone', 'lat', 'lon', 'qty']].copy()
        demand = zone_data[zone_data['action'] == 'demand'][['zone', 'lat', 'lon', 'qty']].copy()
        
        if supply.empty and demand.empty:
            st.info(f"No zones need intervention in this time block (minimum threshold: {min_bikes} bikes)")
        else:
            # Create map
            center_lat = data['stations_top']['lat'].mean()
            center_lon = data['stations_top']['lon'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='cartodbpositron')
            
            # Add station markers (background)
            for _, row in data['stations_top'].iterrows():
                folium.CircleMarker(
                    [row['lat'], row['lon']],
                    radius=2,
                    color='#cccccc',
                    fill=True,
                    fill_opacity=0.3,
                    opacity=0.3
                ).add_to(m)
            
            # Add zone hulls
            zones_shown = list(supply['zone']) + list(demand['zone'])
            if zones_shown:
                hulls = create_zone_hulls_geojson(data['stations_top'], zones_shown)
                
                def style_function(feature):
                    zone = feature['properties']['zone']
                    action = zone_data[zone_data['zone'] == zone]['action'].iloc[0] if zone in zone_data['zone'].values else 'neutral'
                    return {
                        'fillColor': '#a6dfb3' if action == 'supply' else '#f6b0b0',
                        'color': '#2e8b57' if action == 'supply' else '#c0392b',
                        'weight': 1.5,
                        'fillOpacity': 0.3
                    }
                
                folium.GeoJson(hulls, style_function=style_function).add_to(m)
            
            # Greedy matching
            routes = greedy_matching(supply, demand, top_k=max_routes)
            
            if not routes.empty:
                max_bikes = routes['bikes_moved'].max()
                
                for _, route in routes.iterrows():
                    s_zone = centroids[centroids['zone'] == route['supply_zone']].iloc[0]
                    d_zone = centroids[centroids['zone'] == route['demand_zone']].iloc[0]
                    
                    weight = 1 + 6 * (route['bikes_moved'] / max_bikes)
                    folium.PolyLine(
                        [(s_zone['lat'], s_zone['lon']), (d_zone['lat'], d_zone['lon'])],
                        color='#555555',
                        weight=weight,
                        opacity=0.8
                    ).add_to(m)
                    
                    mid_lat = (s_zone['lat'] + d_zone['lat']) / 2
                    mid_lon = (s_zone['lon'] + d_zone['lon']) / 2
                    folium.Marker(
                        [mid_lat, mid_lon],
                        icon=folium.DivIcon(html=f'''
                            <div style="font-weight:700; background:white; padding:2px 6px; 
                                        border-radius:6px; opacity:0.9; font-size:11px;">
                                {format_bikes(route['bikes_moved'])}
                            </div>
                        ''')
                    ).add_to(m)
            
            # Add zone markers
            for _, row in supply.iterrows():
                folium.CircleMarker(
                    [row['lat'], row['lon']],
                    radius=7,
                    color='#2e8b57',
                    fill=True,
                    fill_opacity=0.9,
                    popup=f"Zone {row['zone']}: Supply {int(row['qty'])} bikes"
                ).add_to(m)
            
            for _, row in demand.iterrows():
                folium.CircleMarker(
                    [row['lat'], row['lon']],
                    radius=7,
                    color='#c0392b',
                    fill=True,
                    fill_opacity=0.9,
                    popup=f"Zone {row['zone']}: Demand {int(row['qty'])} bikes"
                ).add_to(m)
            
            # Display map
            st_folium(m, width=1400, height=600)
            
            # Summary statistics
            st.markdown("---")
            st.subheader("Redistribution Recommendations for the selected Month x DoW Pair")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Supply Zones", len(supply), help="Zones with excess bikes")
            with col2:
                st.metric("Demand Zones", len(demand), help="Zones needing bikes")
            with col3:
                st.metric("Routes Suggested", len(routes), help="Number of redistribution routes")
            
            if not routes.empty:
                st.markdown("**Top Redistribution Routes:**")
                display_routes = routes.head(10).copy()
                display_routes['bikes_moved'] = display_routes['bikes_moved'].round().astype(int)
                st.dataframe(
                    display_routes,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "supply_zone": "From Zone",
                        "demand_zone": "To Zone",
                        "bikes_moved": st.column_config.NumberColumn("Bikes to Move", format="%d")
                    }
                )

except FileNotFoundError:
    st.error(f"File not found: {file_path}")
    st.info("Please update the file path in the sidebar.")
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please check your data file and try again.")