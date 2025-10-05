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
from pathlib import Path
from io import BytesIO
import requests
import re

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Waterfront Expansion Analysis", page_icon="🌊", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: accept local paths OR Google Drive links (auto-convert & download)
# ──────────────────────────────────────────────────────────────────────────────

def gdrive_to_direct(url_or_id: str) -> str:
    """
    Accepts:
      - A normal Drive 'view' URL (…/file/d/<ID>/view?usp=sharing)
      - A direct URL (uc?export=download&id=<ID>)
      - Just the raw <ID>
    Returns a direct-download URL.
    """
    s = url_or_id.strip()
    # Raw file ID?
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        return f"https://drive.google.com/uc?export=download&id={s}"
    # Already a direct link?
    if "uc?export=download&id=" in s:
        return s
    # Common view link
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", s)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return s  # Fall back; may be a normal http(s) URL

def download_url(url: str) -> BytesIO:
    """
    Streams a file from a URL. Handles Google Drive large-file confirm flow.
    Returns a BytesIO ready for pandas.
    """
    with requests.Session() as s:
        r = s.get(url, stream=True)
        # Detect Google Drive warning/confirmation page
        if "drive.google.com" in url and "confirm=" not in r.url:
            for k, v in r.cookies.items():
                if k.startswith("download_warning"):
                    url2 = url + ("&" if "?" in url else "?") + f"confirm={v}"
                    r = s.get(url2, stream=True)
                    break
        r.raise_for_status()
        buf = BytesIO()
        for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
            if chunk:
                buf.write(chunk)
        buf.seek(0)
        return buf

    def load_csv_path_or_drive(input_str: str):
    """
    Return a path (str) if local file exists, otherwise a BytesIO downloaded
    from Google Drive/URL. Caller should pass this to pd.read_csv().
    """
    p = Path(input_str)
    if p.exists():
        return str(p)  # path for pandas
    url = gdrive_to_direct(input_str)
    with st.spinner("Downloading data from Google Drive…"):
        return download_url(url)  # BytesIO for pandas


# ──────────────────────────────────────────────────────────────────────────────
# TITLE & INTRO
# ──────────────────────────────────────────────────────────────────────────────

st.title("Q2: How could we determine how many more stations to add along the water?")

st.markdown("""
We see the question on whether to expand on the waterfront as a capacity planning question we can
answer by comparing **supply to demand**:

- **Supply**: share of stations within 300m of water (waterfront buffer)
- **Demand**: share of trip **endpoints** (starts + ends) at waterfront stations

If demand share > supply share, the waterfront is under-served.
""")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (local path OR Drive link/ID)
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Configuration")

trips_input = st.sidebar.text_input(
    "Trips CSV (local path or Google Drive link/ID)",
    value=r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data\nyc_2022_essential_data.csv",
    help="Paste a local path or a Google Drive sharing link (or the raw file ID).",
)

station_input = st.sidebar.text_input(
    "station_to_nta.csv (local path or Google Drive link/ID)",
    value=r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data\station_to_nta.csv",
    help="Paste a local path or a Google Drive sharing link (or the raw file ID).",
)

# ──────────────────────────────────────────────────────────────────────────────
# OSM / waterfront buffer
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_waterfront_buffer():
    """Get the 300m land-side waterfront buffer using OSM (EPSG:26918 meters)."""
    METRIC = "EPSG:26918"  # UTM 18N, meters
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

    water_poly_m = safe_features_m(poly_4326, {"natural": "water"})
    coastline_m = safe_features_m(poly_4326, {"natural": "coastline"})

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

    land = nyc_poly_m.difference(water_union.buffer(1))
    ribbon_land = water_union.buffer(300).intersection(land)
    return ribbon_land, METRIC

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_station_data_any(input_str):
    """Load station_to_nta.csv from local path or Drive; flag waterfront via OSM buffer."""
    station_to_nta = load_csv_path_or_drive(input_str)

    s = station_to_nta.copy()
    s["station_key"] = s["station_name"].astype("string").str.strip().str.lower()

    def _mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else x.iloc[0]

    unique = (
        s.groupby("station_key", as_index=False)
         .agg(station_name=("station_name", "first"),
              lat=("lat", "median"),
              lng=("lng", "median"),
              borough=("borough", _mode))
    )

    ribbon_land, METRIC = get_waterfront_buffer()

    stations_gdf = gpd.GeoDataFrame(
        unique,
        geometry=gpd.points_from_xy(unique["lng"], unique["lat"]),
        crs="EPSG:4326"
    ).to_crs(METRIC)

    buffer_gdf = gpd.GeoDataFrame({"geometry": [ribbon_land]}, crs=METRIC)

    stations_in_buffer = gpd.sjoin(stations_gdf, buffer_gdf, how="left", predicate="within")
    stations_gdf["near_water"] = stations_in_buffer.index_right.notna()

    return stations_gdf[["station_key", "station_name", "lat", "lng", "borough", "near_water"]].copy()

@st.cache_data(show_spinner=False)
def calculate_endpoint_shares_any(trips_input_str, station_flags):
    """Supply/demand shares using local OR Drive trips CSV."""
    trips_df_iter = pd.read_csv(load_csv_path_or_drive(trips_input_str), usecols=["start_station_name", "end_station_name"], chunksize=500_000, dtype="string", low_memory=False)
    flag_map = dict(zip(station_flags['station_key'], station_flags['near_water']))

    supply_share = float(station_flags['near_water'].mean())
    total_stations = int(len(station_flags))

    endpoints_water, total_trips = 0, 0
    for chunk in trips_df_iter:
        s = chunk["start_station_name"].str.strip().str.lower().map(flag_map).fillna(False)
        e = chunk["end_station_name"].str.strip().str.lower().map(flag_map).fillna(False)
        endpoints_water += int(s.sum() + e.sum())
        total_trips += len(chunk)

    demand_share = endpoints_water / (2 * total_trips) if total_trips else 0.0
    shortfall = max(0, int(round((demand_share - supply_share) * total_stations)))
    return {
        "supply_share": supply_share,
        "demand_share": demand_share,
        "total_stations": total_stations,
        "waterfront_stations": int(station_flags['near_water'].sum()),
        "shortfall": shortfall,
        "total_trips": total_trips,
        "waterfront_endpoints": endpoints_water,
    }

@st.cache_data(show_spinner=False)
def calculate_station_activity_any(trips_input_str, station_flags):
    """Trip counts per waterfront station from local OR Drive trips CSV."""
    water_keys = set(station_flags[station_flags['near_water']]['station_key'])
    counts = Counter()

    for chunk in pd.read_csv(load_csv_path_or_drive(trips_input_str),
                             usecols=["start_station_name", "end_station_name"],
                             dtype="string", chunksize=500_000, low_memory=False):
        s = chunk["start_station_name"].str.strip().str.lower()
        e = chunk["end_station_name"].str.strip().str.lower()
        counts.update(s[s.isin(water_keys)].value_counts().to_dict())
        counts.update(e[e.isin(water_keys)].value_counts().to_dict())

    wf_counts = pd.DataFrame({"station_key": list(counts.keys()), "endpoints": list(counts.values())})
    wf_counts = wf_counts.merge(station_flags[['station_key', 'station_name', 'borough']], on='station_key', how='left')
    wf_counts = wf_counts.sort_values('endpoints', ascending=False).reset_index(drop=True)
    wf_counts['share_of_water_endpoints'] = wf_counts['endpoints'] / wf_counts['endpoints'].sum()
    wf_counts['cum_share'] = wf_counts['share_of_water_endpoints'].cumsum()
    return wf_counts

@st.cache_data(show_spinner=False)
def calculate_hourly_patterns_any(trips_input_str, station_flags, wf_counts):
    """Hourly demand patterns for top waterfront hotspots (50% of endpoints)."""
    hotspots = wf_counts[wf_counts['cum_share'] <= 0.50].copy()
    hotspots['station_key'] = hotspots['station_name'].astype("string").str.strip().str.lower()
    hotspot_keys = set(hotspots['station_key'])
    key_to_name = dict(zip(hotspots['station_key'], hotspots['station_name']))

    hour_counts = {k: np.zeros(24, dtype=np.int64) for k in hotspot_keys}

    for chunk in pd.read_csv(load_csv_path_or_drive(trips_input_str),
                             usecols=["start_station_name", "end_station_name", "started_at", "ended_at"],
                             dtype="string", chunksize=500_000, low_memory=False):
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
            pd.Index(e_key[e_key.isin(hotspot_keys)].unique())
        )

        for k in involved:
            ms = (s_key == k)
            me = (e_key == k) & (~ms)
            if ms.any():
                hour_counts[k] += np.bincount(s_hr[ms].dropna().astype(int), minlength=24)
            if me.any():
                hour_counts[k] += np.bincount(e_hr[me].dropna().astype(int), minlength=24)

    rows = []
    for k, counts in hour_counts.items():
        total = counts.sum()
        if total == 0:
            continue
        p = counts / total
        rows.append({
            "station_name": key_to_name.get(k, k),
            "total_trips": int(total),
            "peak_hour": int(np.argmax(counts)),
            "peak_share": float(counts.max() / total),
            "top3_share": float(np.sort(p)[-3:].sum()),
            "span50_hours": int((np.sort(p)[::-1].cumsum() <= 0.5).sum() + 1),
        })
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

try:
    with st.spinner("Loading station catalog & building waterfront buffer…"):
        stations = load_station_data_any(station_input)

    waterfront_count = int(stations['near_water'].sum())
    st.success(f"✓ Loaded {len(stations)} unique stations "
               f"({waterfront_count} waterfront, {len(stations) - waterfront_count} inland)")

    with st.spinner("Analyzing trip patterns (this can take a while for large CSVs)…"):
        metrics = calculate_endpoint_shares_any(trips_input, stations)
        wf_counts = calculate_station_activity_any(trips_input, stations)
        hourly_data = calculate_hourly_patterns_any(trips_input, stations, wf_counts)

    # ── Supply vs Demand
    st.markdown("---")
    st.subheader("Supply vs. Demand Analysis")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Stations", f"{metrics['total_stations']:,}")
    with c2:
        st.metric("Waterfront Stations", f"{metrics['waterfront_stations']}",
                  delta=f"{metrics['supply_share']:.1%}")
    with c3:
        st.metric("Waterfront Endpoints", f"{metrics['waterfront_endpoints']:,}",
                  delta=f"{metrics['demand_share']:.1%}")
    with c4:
        label = "Station Gap" if metrics['shortfall'] > 0 else "No Gap"
        st.metric(label, f"{metrics['shortfall']}",
                  delta=f"{(metrics['demand_share'] - metrics['supply_share']):.1%}",
                  delta_color="inverse" if metrics['shortfall'] > 0 else "off")

    if metrics['shortfall'] == 0:
        st.success("**✅ No System-Wide Waterfront Shortfall** — supply roughly matches demand.")
    else:
        st.warning(f"**⚠️ Potential Shortfall** — implied shortfall ≈ {metrics['shortfall']} stations.")

    # ── Concentration
    st.markdown("---")
    st.subheader("Waterfront Station Concentration")

    top10_share = wf_counts['share_of_water_endpoints'].head(10).sum()
    top10pct_n = max(1, int(round(0.10 * len(wf_counts))))
    top10pct_share = wf_counts['share_of_water_endpoints'].head(top10pct_n).sum()
    hotspots_n = int((wf_counts['cum_share'] <= 0.50).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Top 10 Share", f"{top10_share:.1%}")
    c2.metric(f"Top 10% ({top10pct_n}) Share", f"{top10pct_share:.1%}")
    c3.metric("Hotspot Count (50%)", hotspots_n)

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
    fig.add_bar(name="Top 10", x=[seg1], y=[""], orientation="h", marker_color="#8B0000",
                text=["Top 10 stations"], textposition="inside", insidetextanchor="middle",
                textfont_color="white", hovertemplate="Share: %{x:.1%}<extra></extra>")
    fig.add_bar(name="Top 10%", x=[seg2], y=[""], orientation="h", marker_color="#B22222",
                text=["Top 10% stations"], textposition="inside", insidetextanchor="middle",
                textfont_color="white",
                hovertemplate=f"Next {counts['seg2']} stations<br>Added share: %{{x:.1%}}<extra></extra>")
    fig.add_bar(name=f"Top {hotspots_n}", x=[seg3], y=[""], orientation="h", marker_color="#DC6E6E",
                text=[f"Top {hotspots_n} stations"], textposition="inside", insidetextanchor="middle",
                textfont_color="white",
                hovertemplate=f"Next {counts['seg3']} stations<br>Added share: %{{x:.1%}}<extra></extra>")
    fig.add_bar(name="Remaining", x=[seg4], y=[""], orientation="h", marker_color="#F4CACA",
                text=[f"Remaining {counts['seg4']} stations"], textposition="inside", hoverinfo="skip")
    fig.update_layout(barmode="stack", height=200, margin=dict(l=30, r=20, t=40, b=40),
                      title="Distribution of Endpoints among Waterfront Stations",
                      showlegend=False,
                      xaxis=dict(range=[0, 1], tickformat=".0%", tickvals=[0, 0.5, 1.0],
                                 ticktext=["0%", "50%", "100%"], title=None),
                      yaxis=dict(showticklabels=False, title=None))
    st.plotly_chart(fig, use_container_width=True)

    # ── Timing concentration & candidates
    st.markdown("---")
    st.subheader("Timing Concentration & Expansion Candidates")

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
                        opacity=0.95 if group == "Candidate" else 0.75),
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
        hoverlabel=dict(bgcolor="white"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Recommendations
    st.markdown("---")
    st.subheader("Recommendations")

    candidate_count = int(cand_mask.sum())
    st.markdown(f"""
**With 300m definition and station-count supply:**
- System balance: endpoint share ({metrics['demand_share']:.1%}) ≈ station share ({metrics['supply_share']:.1%})
- Concentration: {hotspots_n} stations account for 50% of waterfront endpoints
- Hotspot intensity: Top 10 stations = {top10_share:.1%} of all waterfront activity
- Expansion candidates: **{candidate_count}** stations show concentrated peak-hour demand
""")

    st.download_button(
        "Download Waterfront Station Data (CSV)",
        data=wf_counts.to_csv(index=False),
        file_name="waterfront_stations_analysis.csv",
        mime="text/csv",
    )

except FileNotFoundError as e:
    st.error(f"File not found: {getattr(e, 'filename', str(e))}")
    st.info("Update the inputs in the left sidebar (local path or Google Drive link/ID).")
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.info("Please check your data files and try again.")
