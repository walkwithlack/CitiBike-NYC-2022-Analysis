# Q3 — Zone-based Restocking

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import MultiPoint
import warnings
from _paths import csv_path

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Q3: How to approach Station Restocking in NYC?", page_icon="🔄", layout="wide")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
SMOOTH_WINDOW = 3
MIN_ABS_C = 6

# ==================================================================================
# HELPERS
# ==================================================================================
def month_fmt(x):
    try:
        if isinstance(x, (int, np.integer)):
            return pd.Timestamp(2022, int(x), 1).strftime('%B')
        ts = pd.to_datetime(str(x), errors="coerce")
        if pd.isna(ts):
            return str(x)
        return ts.strftime('%B %Y')
    except Exception:
        return str(x)

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def greedy_matching(supply_zones, demand_zones, top_k=15):
    if supply_zones.empty or demand_zones.empty:
        return pd.DataFrame(columns=['supply_zone', 'demand_zone', 'bikes_moved'])
    supply, demand, routes = supply_zones.copy(), demand_zones.copy(), []
    while not supply.empty and not demand.empty and supply['qty'].sum() > 0 and demand['qty'].sum() > 0:
        min_dist, best_s_idx, best_d_idx = float('inf'), None, None
        for s_idx, s_row in supply.iterrows():
            if s_row['qty'] <= 0: continue
            for d_idx, d_row in demand.iterrows():
                if d_row['qty'] <= 0: continue
                dist = haversine_distance(s_row['lat'], s_row['lon'], d_row['lat'], d_row['lon'])
                if dist < min_dist:
                    min_dist, best_s_idx, best_d_idx = dist, s_idx, d_idx
        if best_s_idx is None or best_d_idx is None: break
        bikes_to_move = min(supply.loc[best_s_idx, 'qty'], demand.loc[best_d_idx, 'qty'])
        routes.append({'supply_zone': supply.loc[best_s_idx, 'zone'],
                       'demand_zone': demand.loc[best_d_idx, 'zone'],
                       'bikes_moved': bikes_to_move})
        supply.loc[best_s_idx, 'qty'] -= bikes_to_move
        demand.loc[best_d_idx, 'qty'] -= bikes_to_move
        if len(routes) >= top_k: break
    return pd.DataFrame(routes).sort_values('bikes_moved', ascending=False) if routes else pd.DataFrame(columns=['supply_zone','demand_zone','bikes_moved'])

def format_bikes(n): return f"{n/1000:.1f}k" if n >= 1000 else str(int(round(n)))

def create_zone_hulls_geojson(stations_df, zones_to_show):
    features = []
    for zone in zones_to_show:
        zone_stations = stations_df[stations_df['geo_zone'] == zone]
        if len(zone_stations) >= 3:
            points = [(row['lon'], row['lat']) for _, row in zone_stations.iterrows()]
            hull = MultiPoint(points).convex_hull
            features.append({'type': 'Feature','properties': {'zone': int(zone)},'geometry': hull.__geo_interface__})
    return {'type': 'FeatureCollection', 'features': features}

# ==================================================================================
# DATA LOADING (repo-relative)
# ==================================================================================
@st.cache_data
def load_data():
    stations = pd.read_csv(csv_path("stations_metadata.csv"))
    flows = pd.read_csv(csv_path("hourly_flows.csv"))
    flows = flows.sort_values(['month', 'dow', 'zone', 'hour'])
    flows['C'] = flows['ends'] - flows['starts']
    flows['C_smooth'] = (flows.groupby(['month', 'dow', 'zone'])['C']
                         .transform(lambda s: s.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()))
    return stations, flows

# ==================================================================================
# APP
# ==================================================================================
st.title("Q3: How to approach Station Restocking in NYC?")
st.markdown("---")

st.markdown("""
This model identifies **when and where** to move bikes across 25 geographic zones (top 300 Manhattan stations). 
It detects supply/demand imbalances at specific times and suggests efficient redistribution routes.
""")

with st.expander("📋 Problem Framing", expanded=False):
    st.markdown("""
    **Core Challenge:** Identify stations that are net sources (trip origins) or sinks (destinations) at specific times.
    **Why zones?** Vans face road constraints; zones keep moves short and practical.
    """)

with st.expander("🔧 Solution Approach (8 Steps)", expanded=False):
    st.markdown("""
    K-means zones → block detection → consolidation → remove self-correcting residential patterns → pressure scaling →
    sticky-zone penalty → greedy nearest-neighbor matching → user controls (β, thresholds, routes).
    """)

with st.expander("⚠️ Model Limitations", expanded=False):
    st.markdown("""
    Historical-only, heuristic thresholds, smoothing blur, greedy (not optimal LP), and consolidation ambiguity.
    """)

st.markdown("---")

# Load data
try:
    with st.spinner("Loading data..."):
        stations_top, baseline_hourly = load_data()
    st.success(f"✓ Loaded {len(stations_top)} stations in {stations_top['geo_zone'].nunique()} zones")
except FileNotFoundError as e:
    st.error(f"Data file not found in repo: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Controls
st.markdown("---")
st.subheader("Interactive Rebalancing Map")

col1, col2, col3 = st.columns(3)
dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

all_months = list(pd.unique(baseline_hourly['month']))
all_dows   = list(pd.unique(baseline_hourly['dow']))

with col1:
    selected_month = st.selectbox("Month", options=all_months, format_func=month_fmt)
with col2:
    selected_dow = st.selectbox("Day of Week", options=all_dows, format_func=lambda x: dow_labels[int(x)])
with col3:
    time_blocks = [
        ("Morning Commute", 5, 9),
        ("Midday", 10, 14),
        ("Evening Commute", 15, 19),
        ("Night", 20, 23),
        ("Overnight", 23, 4),  # wraps past midnight
    ]
    selected_block_label = st.selectbox("Time Block", options=[l for l,_,_ in time_blocks])
    hour_from = [hf for l,hf,_ in time_blocks if l == selected_block_label][0]
    hour_to   = [ht for l,_,ht in time_blocks if l == selected_block_label][0]

st.markdown("**Rebalancing Parameters:**")
col1, col2, col3 = st.columns(3)
with col1:
    beta = st.slider("β (Intervention Intensity)", 0.1, 0.9, 0.5, 0.05)
with col2:
    min_bikes = st.slider("Minimum Bikes Threshold", 0, 50, 10, 5)
with col3:
    max_routes = st.slider("Maximum Routes Shown", 5, 30, 15, 5)

# Filter the baseline window (handle overnight wrap)
if hour_from <= hour_to:
    cond_hours = (baseline_hourly['hour'] >= hour_from) & (baseline_hourly['hour'] < hour_to)
else:
    cond_hours = (baseline_hourly['hour'] >= hour_from) | (baseline_hourly['hour'] < hour_to)

window_data = baseline_hourly[
    (baseline_hourly['month'] == selected_month) &
    (baseline_hourly['dow'] == selected_dow) &
    cond_hours
]

if window_data.empty:
    st.warning("No data available for this selection")
    st.stop()

# Aggregate flows per zone
zone_summary = (window_data.groupby('zone')
               .agg({'C':'sum','starts':'sum','ends':'sum'})
               .reset_index())

zone_summary['total_activity']   = zone_summary['starts'] + zone_summary['ends']
zone_summary['rental_pressure']  = np.where(zone_summary['total_activity']>0, zone_summary['starts']/zone_summary['total_activity'], 0.5)
zone_summary['return_pressure']  = np.where(zone_summary['total_activity']>0, zone_summary['ends']/zone_summary['total_activity'],   0.5)
zone_summary['action']           = np.where(zone_summary['C'] > 0, 'supply', 'demand')
zone_summary['movable'] = np.where(
    zone_summary['action'] == 'supply',
    abs(zone_summary['C']) * zone_summary['return_pressure'],
    abs(zone_summary['C']) * zone_summary['rental_pressure']
)
zone_summary['qty'] = (zone_summary['movable'] * beta).round()

# Thresholds
zone_summary = zone_summary[(zone_summary['qty'] >= min_bikes) & (abs(zone_summary['C']) >= MIN_ABS_C)]

# Merge with zone centroids
centroids = (stations_top.groupby('geo_zone')[['lat','lon']]
            .mean().reset_index().rename(columns={'geo_zone':'zone'}))
zone_data = zone_summary.merge(centroids, on='zone', how='left')

supply = zone_data[zone_data['action']=='supply'][['zone','lat','lon','qty']].copy()
demand = zone_data[zone_data['action']=='demand'][['zone','lat','lon','qty']].copy()

if supply.empty and demand.empty:
    st.info(f"No zones need intervention in this time block (minimum threshold: {min_bikes} bikes)")
    st.stop()

# Map
center_lat, center_lon = stations_top['lat'].mean(), stations_top['lon'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='cartodbpositron')

# background station points
for _, row in stations_top.iterrows():
    folium.CircleMarker([row['lat'], row['lon']], radius=2, color='#cccccc',
                        fill=True, fill_opacity=0.3, opacity=0.3).add_to(m)

# zone hulls
zones_shown = list({*supply['zone'].tolist(), *demand['zone'].tolist()})
if zones_shown:
    hulls = create_zone_hulls_geojson(stations_top, zones_shown)
    def style_function(feature):
        z = feature['properties']['zone']
        action = zone_data.loc[zone_data['zone']==z, 'action']
        action = action.iloc[0] if not action.empty else 'neutral'
        return {'fillColor': '#a6dfb3' if action=='supply' else '#f6b0b0',
                'color': '#2e8b57' if action=='supply' else '#c0392b',
                'weight': 1.5, 'fillOpacity': 0.3}
    folium.GeoJson(hulls, style_function=style_function).add_to(m)

# routes
routes = greedy_matching(supply, demand, top_k=max_routes)
if not routes.empty:
    max_bikes = routes['bikes_moved'].max()
    for _, r in routes.iterrows():
        s = centroids[centroids['zone']==r['supply_zone']].iloc[0]
        d = centroids[centroids['zone']==r['demand_zone']].iloc[0]
        weight = 1 + 6 * (r['bikes_moved'] / max_bikes)
        folium.PolyLine([(s['lat'], s['lon']), (d['lat'], d['lon'])],
                        color='#555555', weight=weight, opacity=0.8).add_to(m)
        mid_lat, mid_lon = (s['lat']+d['lat'])/2, (s['lon']+d['lon'])/2
        folium.Marker([mid_lat, mid_lon],
            icon=folium.DivIcon(html=f'''
                <div style="font-weight:700; background:white; padding:2px 6px;
                            border-radius:6px; opacity:0.9; font-size:11px;">
                    {format_bikes(r['bikes_moved'])}
                </div>
            ''')
        ).add_to(m)

# zone markers
for _, row in supply.iterrows():
    folium.CircleMarker([row['lat'], row['lon']], radius=7, color='#2e8b57',
                        fill=True, fill_opacity=0.9,
                        popup=f"Zone {row['zone']}: Supply {int(row['qty'])} bikes").add_to(m)
for _, row in demand.iterrows():
    folium.CircleMarker([row['lat'], row['lon']], radius=7, color='#c0392b',
                        fill=True, fill_opacity=0.9,
                        popup=f"Zone {row['zone']}: Demand {int(row['qty'])} bikes").add_to(m)

st_folium(m, width=1400, height=600)

# Summary
st.markdown("---")
st.subheader("Redistribution Recommendations for Selected Month × Day-of-Week")
c1, c2, c3 = st.columns(3)
c1.metric("Supply Zones", len(supply))
c2.metric("Demand Zones", len(demand))
c3.metric("Routes Suggested", len(routes))

if not routes.empty:
    st.markdown("**Redistribution Routes:**")
    display_routes = routes.head(10).copy()
    display_routes['bikes_moved'] = display_routes['bikes_moved'].round().astype(int)
    st.dataframe(
        display_routes, use_container_width=True, hide_index=True,
        column_config={
            "supply_zone": "From Zone",
            "demand_zone": "To Zone",
            "bikes_moved": st.column_config.NumberColumn("Bikes to Move", format="%d"),
        },
    )
