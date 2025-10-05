# 01_Introduction.py — Page 1 (auto local-or-GDrive)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from io import BytesIO
import requests
import re

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CitiBike 2022 - Introduction", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Helper: local path OR Google Drive link/ID → readable object for pandas/open()
# ──────────────────────────────────────────────────────────────────────────────
def gdrive_to_direct(url_or_id: str) -> str:
    """Accept raw file ID, 'view' URLs, or uc? links and return a direct-download URL."""
    s = (url_or_id or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):
        return f"https://drive.google.com/uc?export=download&id={s}"
    if "uc?export=download&id=" in s:
        return s
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", s) or re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", s)
    return f"https://drive.google.com/uc?export=download&id={m.group(1)}" if m else s

def download_url(url: str) -> BytesIO:
    """Stream a (possibly large) file; handles Google Drive confirm token."""
    with requests.Session() as s:
        r = s.get(url, stream=True)
        if "drive.google.com" in url and "confirm=" not in r.url:
            for k, v in r.cookies.items():
                if k.startswith("download_warning"):
                    url = url + ("&" if "?" in url else "?") + f"confirm={v}"
                    r = s.get(url, stream=True)
                    break
        r.raise_for_status()
        buf = BytesIO()
        for chunk in r.iter_content(2 * 1024 * 1024):
            if chunk:
                buf.write(chunk)
        buf.seek(0)
        return buf

def resolve_path_or_drive(user_input: str) -> BytesIO | str:
    """Return a local path (str) if it exists; otherwise a BytesIO downloaded from Drive/URL."""
    p = Path(user_input)
    if p.exists():
        return str(p)
    return download_url(gdrive_to_direct(user_input))

def read_csv_safely(path_or_link: str, **kwargs) -> pd.DataFrame:
    src = resolve_path_or_drive(path_or_link)
    return pd.read_csv(src, **kwargs)

# ──────────────────────────────────────────────────────────────────────────────
# Config: local data dir + Drive links as fallback
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data\aggregated")

DRIVE_LINKS = {
    # You provided these IDs/links earlier
    "trip_durations.csv": "https://drive.google.com/file/d/1W3zX7L9_1ht0lU0w_fnSG6UMcWYXPVN5/view?usp=sharing",
    "hourly_patterns.csv": "https://drive.google.com/file/d/18ANpyOF8DX7bPb6wvA3x318YVzf7XkDg/view?usp=sharing",
    "day_of_week_totals.csv": "https://drive.google.com/file/d/1vjdoMn0S2Jsej0Mo1OqrA9tY-y2DnH03/view?usp=sharing",
    "daily_aggregates.csv": "https://drive.google.com/file/d/1mU_QDgo3zeh6ukX8lz7ZCOMk8HCQ5Va3/view?usp=sharing",
    "new_york_citi_bike_map.html": "https://drive.google.com/file/d/1-X65o-K6olM0ZUHZE6OnpW25kqIMwdom/view?usp=sharing",
}

def prefer_local(fname: str) -> str:
    """Return local path if present; otherwise the Drive link for that file."""
    local = DATA_DIR / fname
    return str(local) if local.exists() else DRIVE_LINKS[fname]

# ──────────────────────────────────────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────────────────────────────────────
FLARE_COLORS = ['#e14b31', '#f47e3e', '#f7ad48', '#fbda5f', '#fef574', '#c9e583', '#88cc91', '#4ba99e']
WEEKDAY_COLOR = '#e14b31'
WEEKEND_COLOR = '#4ba99e'

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("NYC CitiBike 2022 Analysis")
st.markdown("## Introduction (Part 1 of 2): Getting Acquainted with the 2022 CitiBike Dataset")
st.markdown("""
Before diving into operational recommendations, let's explore the CitiBike system through 2022 data—understanding 
how people use it, when they ride, and what influences demand. These insights will inform our strategic questions 
about fleet management, expansion, and redistribution.
""")
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Chart 1 — Trip Duration
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Chart 1: How Long Do CitiBike Trips Last?")
st.markdown("*Histogram showing trip duration distribution (0–75 minutes) with KDE overlay*")

@st.cache_data(show_spinner=True)
def load_durations() -> pd.DataFrame:
    return read_csv_safely(prefer_local("trip_durations.csv"))

durations = load_durations()

fig1, ax1 = plt.subplots(figsize=(14, 6))
sns.histplot(durations['tripduration_min'], bins=25, kde=True, color='steelblue', edgecolor='black', ax=ax1)
ax1.set_xlim(0, 75)
ax1.set_xticks(np.arange(0, 76, 3))
ax1.set_xlabel('Trip duration (minutes)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Trip Durations', fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig1)
plt.close()

st.markdown("""
Most trips last between 3 and 9 minutes, indicating CitiBike is often used for short first/last-mile hops.
""")
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Chart 2 — Weekday vs Weekend patterns
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Chart 2: Weekday vs Weekend Usage Patterns")
st.markdown("*Comparison of usage patterns between weekdays and weekends*")

@st.cache_data(show_spinner=True)
def load_hourly_patterns() -> pd.DataFrame:
    return read_csv_safely(prefer_local("hourly_patterns.csv"))

overlay = load_hourly_patterns()

fig2, ax2 = plt.subplots(figsize=(12, 6))
for period, color in [("Weekday", WEEKDAY_COLOR), ("Weekend", WEEKEND_COLOR)]:
    period_data = overlay[overlay["period"] == period]
    ax2.plot(period_data["hour"], period_data["trips_per_day"], label=period, color=color, linewidth=2.5)

ax2.set_title("Weekday vs Weekend: Hourly Trip Patterns", fontsize=14, fontweight='bold')
ax2.set_xlabel("Hour of Day", fontsize=12)
ax2.set_ylabel("Average Trips per Day", fontsize=12)
ax2.set_xticks(range(0, 24, 2))
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("""
Weekdays show sharp commute peaks; weekends are flatter with a broad midday swell.
""")
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Chart 3 — Trips by day of week
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Chart 3: Trip Volume Across the Week")
st.markdown("*Total trips per day of week*")

@st.cache_data(show_spinner=True)
def load_day_of_week() -> pd.DataFrame:
    return read_csv_safely(prefer_local("day_of_week_totals.csv"))

dow_counts = load_day_of_week()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_counts['day_name'] = pd.Categorical(dow_counts['day_name'], categories=dow_order, ordered=True)
dow_counts = dow_counts.sort_values('day_name')

weekend_days = {"Saturday", "Sunday"}
fig3, ax3 = plt.subplots(figsize=(12, 6))
colors = [WEEKDAY_COLOR if day not in weekend_days else WEEKEND_COLOR for day in dow_counts['day_name']]
ax3.bar(range(len(dow_counts)), dow_counts["trips"], color=colors, edgecolor='black', linewidth=0.5)
ax3.set_xticks(range(len(dow_counts)))
ax3.set_xticklabels(dow_counts['day_name'], rotation=45, ha='right')
ax3.set_title("Total Trips per Day of Week", fontsize=14, fontweight='bold')
ax3.set_xlabel("Day of Week", fontsize=12)
ax3.set_ylabel("Total Trips", fontsize=12)
ax3.ticklabel_format(style="plain", axis="y")
ax3.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
st.pyplot(fig3)
plt.close()

st.markdown("""
Weekends keep strong volume—Saturday is close to weekdays; Wed/Thu are the weekday peaks.
""")
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Chart 4 — Weather impact
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Chart 4: Weather's Impact on Ridership")
st.markdown("*Daily bike trips and temperature on dual axis (7-day smoothed)*")

@st.cache_data(show_spinner=True)
def load_daily_aggregates() -> pd.DataFrame:
    df = read_csv_safely(prefer_local("daily_aggregates.csv"))
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['rides_7'] = df['bike_rides_daily'].rolling(7, min_periods=1).mean()
    df['temp_7'] = df['avgTemp'].rolling(7, min_periods=1).mean()
    return df

df_daily = load_daily_aggregates()

fig4, ax4 = plt.subplots(figsize=(14, 6))
ax4.plot(df_daily.index, df_daily['rides_7'], color='blue', lw=2, label='Bike rides (7-day MA)')
ax4.plot(df_daily.index, df_daily['bike_rides_daily'], color='blue', lw=0.7, alpha=0.15)
ax4.set_ylabel('Bike rides', color='blue', fontsize=12)
ax4.tick_params(axis='y', labelcolor='blue')
ax4.spines['left'].set_color('blue')

ax4_2 = ax4.twinx()
ax4_2.plot(df_daily.index, df_daily['temp_7'], color='red', lw=2, label='Avg temp (7-day MA)')
ax4_2.plot(df_daily.index, df_daily['avgTemp'], color='red', lw=0.7, alpha=0.15)
ax4_2.set_ylabel('Temperature (°C)', color='red', fontsize=12)
ax4_2.tick_params(axis='y', labelcolor='red')
ax4_2.spines['right'].set_color('red')

lines, labels = ax4.get_legend_handles_labels()
l2, lab2 = ax4_2.get_legend_handles_labels()
ax4.legend(lines + l2, labels + lab2, loc='upper left', frameon=False)

ax4.grid(True, ls='--', lw=0.5, alpha=0.5)
ax4.set_title('Bike Trips vs Temperature – 2022 (smoothed)', fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig4)
plt.close()

st.markdown("""
Warm days pull ridership up; winter cools the system down—useful for planning fleet scale.
""")
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Chart 5 — Kepler map (large HTML)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Chart 5: The CitiBike Network in Action")
st.markdown("*Map showing highest-volume stations and major origin-destination flows*")

map_src = prefer_local("new_york_citi_bike_map.html")
try:
    src = resolve_path_or_drive(map_src)
    if isinstance(src, str):
        html_data = Path(src).read_text(encoding="utf-8")
    else:
        html_data = src.read().decode("utf-8", "ignore")

    # Warning: very large HTMLs may be heavy to embed on Streamlit Cloud
    st.components.v1.html(html_data, height=1000)
except Exception as e:
    st.info(
        "The Kepler map is very large and couldn’t be embedded here. "
        "You can open it directly from Google Drive:\n\n"
        f"{DRIVE_LINKS['new_york_citi_bike_map.html']}\n\n"
        f"(Error: {e})"
    )

st.markdown("""
This gives the spatial context—bigger/darker stations = higher trip volume; lines = big OD flows.
""")
st.markdown("---")
