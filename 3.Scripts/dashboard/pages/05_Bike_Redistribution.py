"""
Lightweight Bike Redistribution Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import MultiPoint
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Q3: How to approach Station Restocking in NYC?", page_icon="🔄", layout="wide")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
SMOOTH_WINDOW = 3
MIN_ABS_C = 6

# ==================================================================================
# HELPER FUNCTIONS
# ==================================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def greedy_matching(supply_zones, demand_zones, top_k=15):
    """Greedily match supply to demand zones by shortest distance"""
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
                dist = haversine_distance(s_row['lat'], s_row['lon'], d_row['lat'], d_row['lon'])
                if dist < min_dist:
                    min_dist, best_s_idx, best_d_idx = dist, s_idx, d_idx
        
        if best_s_idx is None or best_d_idx is None:
            break
            
        bikes_to_move = min(supply.loc[best_s_idx, 'qty'], demand.loc[best_d_idx, 'qty'])
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
    """Format bike counts for display"""
    return f"{n/1000:.1f}k" if n >= 1000 else str(int(round(n)))

def create_zone_hulls_geojson(stations_df, zones_to_show):
    """Create convex hulls around stations in each zone"""
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

# ==================================================================================
# DATA LOADING
# ==================================================================================

@st.cache_data
def load_data():
    """Load the pre-processed lightweight data files"""
    stations = pd.read_csv(r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data\stations_metadata.csv")
    flows = pd.read_csv(r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data\hourly_flows.csv")
    
    # Calculate smoothed net flow
    flows = flows.sort_values(['month', 'dow', 'zone', 'hour'])
    flows['C'] = flows['ends'] - flows['starts']
    flows['C_smooth'] = (flows.groupby(['month', 'dow', 'zone'])['C']
                         .transform(lambda x: x.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()))
    
    return stations, flows

# ==================================================================================
# STREAMLIT APP
# ==================================================================================

st.title("Q3: How to approach Station Restocking in NYC?")

st.markdown("---")

# Brief overview always visible
st.markdown("""
This model identifies **when and where** to move bikes across 25 geographic zones (top 300 Manhattan stations). 
It detects supply/demand imbalances at specific times and suggests efficient redistribution routes.
""")

# Expandable sections
with st.expander("📋 Problem Framing", expanded=False):
    st.markdown("""
    **Core Challenge:** Identify stations that are net sources (trip origins) or sinks (destinations) at specific times, 
    since bikes must come from somewhere to go elsewhere.
    
    **Multi-scale Patterns:**
    - **Monthly:** Temperature affects total volume
    - **Day-of-week:** Weekday vs. weekend behavior  
    - **Hourly:** Commuter peaks flip stations from morning sinks to evening sources
    
    **Key Insight:** Many stations have bidirectional demand—high both starts *and* ends. We're **not seeking 
    zero net flow**; some "balanced" stations need intervention to maintain capacity for expected demand patterns whereas "unbalanced" stations may be perfect that way at certain times.
    
    **Why Geographic Zones?** Redistribution vans face road constraints. We avoided trip-flow clustering 
    (high-volume pairs aren't necessarily close) and administrative boundaries (users ignore these).
    """)

with st.expander("🔧 Solution Approach (8 Steps)", expanded=False):
    st.markdown("""
    **1. Geographic Zones**  
    K-means clustering (lat/lon) grouped top 300 stations → 25 static zones. Zones remain fixed; their roles shift over time.
    
    **2. Zone-Level Block Detection**  
    For each zone×month×dow, detect time blocks where smoothed net flow (C = ends − starts) stays consistently 
    positive (supply) or negative (demand). Minimum: 2 hours duration, |C| ≥ 6 bikes.
    
    **3. Two-Regime Consolidation**  
    - **Overnight (23:00-04:00):** One canonical block per month×dow by summing hourly C  
    - **Daytime (04:00-23:00):** Merge blocks within 3-hour tolerance  
    Result: 4,644 fragmented blocks → 731 canonical windows (~9 per month×dow)
    
    **4. Self-Correcting Residential Zones**  
    Exclude zones with morning deficits (C < -20), evening surpluses (C > +20), quiet middays (|C| < 15)—
    they balance naturally. Found 45 combinations across 15 zones; excluded 311 block assignments.
    
    **5. Bidirectional Demand Adjustment**  
    Calculate rental pressure (starts ÷ total activity) and return pressure (ends ÷ total activity):
    - Supply zones scale by *return* pressure (preserve rental capacity)  
    - Demand zones scale by *rental* pressure (acknowledge ongoing returns)  
    Reduced movable quantities to 53% of raw |C|—prevents stripping busy stations bare.
    
    **6. Sticky Zones Heuristic**  
    Zones flipping roles between consecutive blocks (supply ↔ demand) get 50% quantity reduction. 
    Identified 2,737 sticky blocks (31.7%), preventing wasteful back-and-forth moves.
    
    **7. Greedy Matching**  
    Match supply → demand by nearest-neighbor distance (Haversine). Move min(supply, demand) per route 
    until exhausted or max routes reached.
    
    **8. User Controls**  
    β parameter (0.1–0.9) scales all quantities (conservative ↔ aggressive). Additional filters for minimum 
    moves and maximum routes. Interactive map shows zones, supply/demand status, and suggested routes—grounded 
    in 2022 patterns and operational constraints.
    """)

with st.expander("⚠️ Model Limitations", expanded=False):
    st.markdown("""
    **1. Historical Data Only**  
    Uses 2022 data without real-time updates. Also missing: traffic, terrain, precipitation, staffing. Therefore, the model treats 2022 data as deterministic forecasts without uncertainty quantification.
    
    **2. Temporal Ambiguity from Consolidation**  
    Merging blocks (±3hr tolerance) reduces fragmentation (4,644→731) but forces zones with different timing onto 
    shared labels, obscuring precise intervention moments. Some consolidated blocks overlap (e.g., 04:00-13:00, 
    05:00-09:00, 08:00-16:00), creating dropdown redundancy. Partially mitigated by focusing on top 300 Manhattan 
    stations with synchronized patterns.
    
    **3. Threshold Sensitivity**  
    Arbitrary cutoffs lack principled derivation: |C| ≥ 6, residential cutoffs, sticky penalty = 50%, β = 0.1-0.9. 
    Small changes significantly alter recommendations without systematic optimization.
    
    **4. Greedy Matching Suboptimality**  
    Nearest-neighbor heuristic is computationally simple but not guaranteed optimal for total route distance or 
    operational efficiency (no linear programming).
    
    **5. Smoothing Temporal Blur**  
    3-hour centered moving average stabilizes regime detection but blurs sharp transitions. A 9:00am shift may 
    appear as gradual 8:30-9:30am, misaligning blocks with operational reality.
    
    ---
    
    *We built this model despite these substantial limitations as a way of engaging with the problem's complexity. For the full analysis 
    with tailored (non-canonical) time blocks, see `NYC_Q3_Zone-Based_Restocking.ipynb`. For data preparation details, 
    see `NYC_Q3_Extraction.ipynb`.*
    """)

st.markdown("---")

# Load data
try:
    with st.spinner("Loading data..."):
        stations_top, baseline_hourly = load_data()
    
    st.success(f"✓ Loaded {len(stations_top)} stations in {stations_top['geo_zone'].nunique()} zones")
    
    # Controls
    st.markdown("---")
    st.subheader("Interactive Rebalancing Map")
    
    col1, col2, col3 = st.columns(3)
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    all_months = sorted(baseline_hourly['month'].unique())
    all_dows = sorted(baseline_hourly['dow'].unique())
    
    with col1:
        selected_month = st.selectbox(
            "Month", 
            options=all_months,
            format_func=lambda x: pd.Period(x).strftime('%B %Y')
        )
    
    with col2:
        selected_dow = st.selectbox(
            "Day of Week", 
            options=all_dows,
            format_func=lambda x: dow_labels[x]
        )
    
    with col3:
        time_blocks = [
            ("Morning Commute", 5, 9),
            ("Midday", 10, 14),
            ("Evening Commute", 15, 19),
            ("Night", 20, 23),
            ("Overnight", 23, 4)
        ]
        selected_block_label = st.selectbox("Time Block", options=[l for l, _, _ in time_blocks])
        hour_from = [hf for l, hf, _ in time_blocks if l == selected_block_label][0]
        hour_to = [ht for l, _, ht in time_blocks if l == selected_block_label][0]
    
    st.markdown("**Rebalancing Parameters:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        beta = st.slider(
            "β (Intervention Intensity)", 
            0.1, 0.9, 0.5, 0.05,
            help="Fraction of movable bikes to move (0.1=conservative, 0.9=aggressive)"
        )
    
    with col2:
        min_bikes = st.slider(
            "Minimum Bikes Threshold", 
            0, 50, 10, 5,
            help="Hide zones with recommendations below this threshold"
        )
    
    with col3:
        max_routes = st.slider(
            "Maximum Routes Shown", 
            5, 30, 15, 5,
            help="Limit number of supply→demand routes displayed"
        )
    
    # Filter the baseline window
    window_data = baseline_hourly[
        (baseline_hourly['month'] == selected_month) &
        (baseline_hourly['dow'] == selected_dow) &
        (baseline_hourly['hour'] >= hour_from) &
        (baseline_hourly['hour'] < hour_to)
    ]
    
    if window_data.empty:
        st.warning("No data available for this selection")
    else:
        # Aggregate flows per zone
        zone_summary = (window_data.groupby('zone')
                       .agg({'C': 'sum', 'starts': 'sum', 'ends': 'sum'})
                       .reset_index())
        
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
        
        # Calculate action and movable quantities
        zone_summary['action'] = np.where(zone_summary['C'] > 0, 'supply', 'demand')
        zone_summary['movable'] = np.where(
            zone_summary['action'] == 'supply',
            abs(zone_summary['C']) * zone_summary['return_pressure'],
            abs(zone_summary['C']) * zone_summary['rental_pressure']
        )
        zone_summary['qty'] = (zone_summary['movable'] * beta).round()
        
        # Filter by thresholds
        zone_summary = zone_summary[
            (zone_summary['qty'] >= min_bikes) & 
            (abs(zone_summary['C']) >= MIN_ABS_C)
        ]
        
        # Merge with station coordinates (zone centroids)
        centroids = (stations_top.groupby('geo_zone')[['lat', 'lon']]
                    .mean()
                    .reset_index()
                    .rename(columns={'geo_zone': 'zone'}))
        
        zone_data = zone_summary.merge(centroids, on='zone', how='left')
        
        supply = zone_data[zone_data['action'] == 'supply'][['zone', 'lat', 'lon', 'qty']].copy()
        demand = zone_data[zone_data['action'] == 'demand'][['zone', 'lat', 'lon', 'qty']].copy()
        
        if supply.empty and demand.empty:
            st.info(f"No zones need intervention in this time block (minimum threshold: {min_bikes} bikes)")
        else:
            # Create map
            center_lat = stations_top['lat'].mean()
            center_lon = stations_top['lon'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='cartodbpositron')
            
            # Background station points
            for _, row in stations_top.iterrows():
                folium.CircleMarker(
                    [row['lat'], row['lon']], 
                    radius=2, 
                    color='#cccccc',
                    fill=True, 
                    fill_opacity=0.3, 
                    opacity=0.3
                ).add_to(m)
            
            # Zone hulls
            zones_shown = list(supply['zone']) + list(demand['zone'])
            if zones_shown:
                hulls = create_zone_hulls_geojson(stations_top, zones_shown)
                
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
            
            # Routes
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
            
            # Zone markers
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
            
            st_folium(m, width=1400, height=600)
            
            # Summary
            st.markdown("---")
            st.subheader("Redistribution Recommendations for Selected Month x Day-of-Week Pair")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Supply Zones", len(supply))
            c2.metric("Demand Zones", len(demand))
            c3.metric("Routes Suggested", len(routes))
            
            if not routes.empty:
                st.markdown("**Redistribution Routes:**")
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
    st.error("Data files not found!")
    st.info("""
    Please make sure you have these files in the same directory as this script:
    - `stations_metadata.csv`
    - `hourly_flows.csv`
    
    Run the data extraction script first to generate these files from your large CSV.
    """)
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please check your data files and try again.")