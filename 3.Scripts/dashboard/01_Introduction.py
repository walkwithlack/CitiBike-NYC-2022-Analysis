import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page config
st.set_page_config(page_title="CitiBike 2022 - Introduction", layout="wide")

# Color scheme
FLARE_COLORS = ['#e14b31', '#f47e3e', '#f7ad48', '#fbda5f', '#fef574', '#c9e583', '#88cc91', '#4ba99e']
WEEKDAY_COLOR = '#e14b31'
WEEKEND_COLOR = '#4ba99e'

# Paths
DATA_DIR = Path(r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data\aggregated")

# Header
st.title("NYC CitiBike 2022 Analysis")
st.markdown("## Introduction (Part 1 of 2): Getting Acquainted with the 2022 CitiBike Dataset")
st.markdown("""
Before diving into operational recommendations, let's explore the CitiBike system through 2022 data—understanding 
how people use it, when they ride, and what influences demand. These insights will inform our strategic questions 
about fleet management, expansion, and redistribution.
""")

st.markdown("---")

# ==================== CHART 1: Trip Duration ====================
st.markdown("### Chart 1: How Long Do CitiBike Trips Last?")
st.markdown("*Histogram showing trip duration distribution (0-75 minutes) with KDE overlay*")

@st.cache_data
def load_durations():
    return pd.read_csv(DATA_DIR / "trip_durations.csv")

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
Most trips last between 3 and 6 minutes, followed by trips of 6-9 minutes. 
This indicates that CitiBike is mostly used as a first- or last-mile mobility solution.
""")

st.markdown("---")

# ==================== CHART 2: Weekday vs Weekend ====================
st.markdown("### Chart 2: Weekday vs Weekend Usage Patterns")
st.markdown("*Comparison of usage patterns between weekdays and weekends*")

@st.cache_data
def load_hourly_patterns():
    return pd.read_csv(DATA_DIR / "hourly_patterns.csv")

overlay = load_hourly_patterns()

fig2, ax2 = plt.subplots(figsize=(12, 6))
for period, color in [("Weekday", WEEKDAY_COLOR), ("Weekend", WEEKEND_COLOR)]:
    period_data = overlay[overlay["period"] == period]
    ax2.plot(period_data["hour"], period_data["trips_per_day"], 
             label=period, color=color, linewidth=2.5)

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
We see that there are two major usage modes in the system—commuter-driven weekday patterns with sharp morning and evening peaks versus leisure-focused weekend riding with a smoother daytime distribution.
""")

st.markdown("---")

# ==================== CHART 3: Trip Volume by Day of Week ====================
st.markdown("### Chart 3: Trip Volume Across the Week")
st.markdown("*Total trips per day of week*")

@st.cache_data
def load_day_of_week():
    return pd.read_csv(DATA_DIR / "day_of_week_totals.csv")

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
The chart challenges an intuitive assumption by showing that weekends maintain substantial volume, not the dramatic drop-off one might expect. Saturday approaches weekday levels, and mid-week (Wednesday/Thursday) shows peak activity.
""")

st.markdown("---")

# ==================== CHART 4: Weather Impact ====================
st.markdown("### Chart 4: Weather's Impact on Ridership")
st.markdown("*Daily bike trips and temperature on dual axis (7-day smoothed)*")

@st.cache_data
def load_daily_aggregates():
    df = pd.read_csv(DATA_DIR / "daily_aggregates.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['rides_7'] = df['bike_rides_daily'].rolling(7, min_periods=1).mean()
    df['temp_7'] = df['avgTemp'].rolling(7, min_periods=1).mean()
    return df

df_daily = load_daily_aggregates()

fig4, ax4 = plt.subplots(figsize=(14, 6))

# Left axis: bike rides (blue)
ax4.plot(df_daily.index, df_daily['rides_7'], color='blue', lw=2, label='Bike rides (7-day MA)')
ax4.plot(df_daily.index, df_daily['bike_rides_daily'], color='blue', lw=0.7, alpha=0.15)
ax4.set_ylabel('Bike rides', color='blue', fontsize=12)
ax4.tick_params(axis='y', labelcolor='blue')
ax4.spines['left'].set_color('blue')

# Right axis: temperature (red)
ax4_2 = ax4.twinx()
ax4_2.plot(df_daily.index, df_daily['temp_7'], color='red', lw=2, label='Avg temp (7-day MA)')
ax4_2.plot(df_daily.index, df_daily['avgTemp'], color='red', lw=0.7, alpha=0.15)
ax4_2.set_ylabel('Temperature (°C)', color='red', fontsize=12)
ax4_2.tick_params(axis='y', labelcolor='red')
ax4_2.spines['right'].set_color('red')

# Combined legend
lines, labels = ax4.get_legend_handles_labels()
l2, lab2 = ax4_2.get_legend_handles_labels()
ax4.legend(lines + l2, labels + lab2, loc='upper left', frameon=False)

ax4.grid(True, ls='--', lw=0.5, alpha=0.5)
ax4.set_title('Bike Trips vs Temperature – 2022 (smoothed)', fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig4)
plt.close()

st.markdown("""
The chart visualizes a strong correlation between temperature and bike usage—a critical factor for operational planning. The smoothed lines reveal the underlying seasonal trend while keeping daily fluctuations visible.
""")

st.markdown("---")

# ==================== CHART 5: Kepler Map ====================
st.markdown("### Chart 5: The CitiBike Network in Action")
st.markdown("*Map showing highest-volume stations and major origin-destination flows*")

path_to_html = r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\4.Visualizations\new_york_citi_bike_map.html"

with open(path_to_html, 'r', encoding='utf-8') as f: 
    html_data = f.read()

st.components.v1.html(html_data, height=1000)

st.markdown("""
This is the spatial context for where demand concentrates and how bikes move through the city—the geographic foundation for understanding expansion and redistribution needs. Larger/darker stations indicate higher trip volumes, while connecting lines show major origin-destination flows.
""")

st.markdown("---")

