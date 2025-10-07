# 01_Introduction.py — Clean version with interactive charts

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CitiBike 2022 - Introduction", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Data paths (relative for GitHub deployment)
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("C:/Users/magia/OneDrive/Desktop/NY_Citi_Bike/2.Data/Prepared Data")
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────────────────────────────────────
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

**Note**: This page uses pre-processed files. To check their creation, see NYC_01_Intro_Preprocessing.ipynb.
""")
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Chart 1 — Trip Duration (with matching color scheme)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Chart 1: How Long Do CitiBike Trips Last?")
st.markdown("*Histogram showing trip duration distribution (0–75 minutes) with KDE overlay*")

@st.cache_data(show_spinner=True)
def load_durations() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR/"trip_durations.csv")

durations = load_durations()

fig1, ax1 = plt.subplots(figsize=(14, 6))

# Plot histogram without KDE
sns.histplot(durations['tripduration_min'], bins=25, kde=False, 
             color=WEEKEND_COLOR, edgecolor='black', ax=ax1, stat='count')

# Create a second y-axis for the KDE
ax2 = ax1.twinx()
sns.kdeplot(durations['tripduration_min'], color=WEEKDAY_COLOR, 
            linewidth=2.5, ax=ax2)
ax2.set_ylabel('')  # Hide the second y-axis label
ax2.set_yticks([])  # Hide the second y-axis ticks

ax1.set_xlim(0, 75)
ax1.set_xticks(np.arange(0, 76, 3))
ax1.set_xlabel('Trip duration (minutes)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Trip Durations', fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig1)
plt.close()

st.markdown("""
Most trips last between 3 and 9 minutes, indicating **CitiBike is often used for short first/last-mile hops**.
""")
st.markdown("---")
# ──────────────────────────────────────────────────────────────────────────────
# Chart 2 — Weekday vs Weekend patterns (PLOTLY)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Chart 2: Weekday vs Weekend Usage Patterns")
st.markdown("*Comparison of usage patterns between weekdays and weekends*")

@st.cache_data(show_spinner=True)
def load_hourly_patterns() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR/"hourly_patterns.csv")

overlay = load_hourly_patterns()

fig2 = go.Figure()

weekday_data = overlay[overlay["period"] == "Weekday"]
weekend_data = overlay[overlay["period"] == "Weekend"]

fig2.add_trace(go.Scatter(
    x=weekday_data["hour"],
    y=weekday_data["trips_per_day"],
    mode='lines',
    name='Weekday',
    line=dict(color=WEEKDAY_COLOR, width=3),
    hovertemplate='Hour: %{x}<br>Trips/day: %{y:,.0f}<extra></extra>'
))

fig2.add_trace(go.Scatter(
    x=weekend_data["hour"],
    y=weekend_data["trips_per_day"],
    mode='lines',
    name='Weekend',
    line=dict(color=WEEKEND_COLOR, width=3),
    hovertemplate='Hour: %{x}<br>Trips/day: %{y:,.0f}<extra></extra>'
))

fig2.update_layout(
    title="Weekday vs Weekend: Hourly Trip Patterns",
    xaxis_title="Hour of Day",
    yaxis_title="Average Trips per Day",
    hovermode='x unified',
    height=500,
    xaxis=dict(tickmode='linear', tick0=0, dtick=2)
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
**Weekdays show sharp commute peaks**; **weekends are flatter with a broad midday swell**.
""")
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Chart 3 — Trips by day of week (PLOTLY)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Chart 3: Trip Volume Across the Week")
st.markdown("*Total trips per day of week*")

@st.cache_data(show_spinner=True)
def load_day_of_week() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR/"day_of_week_totals.csv")

dow_counts = load_day_of_week()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_counts['day_name'] = pd.Categorical(dow_counts['day_name'], categories=dow_order, ordered=True)
dow_counts = dow_counts.sort_values('day_name')

weekend_days = {"Saturday", "Sunday"}
dow_counts['color'] = dow_counts['day_name'].apply(
    lambda x: WEEKEND_COLOR if x in weekend_days else WEEKDAY_COLOR
)

fig3 = go.Figure(data=[
    go.Bar(
        x=dow_counts['day_name'],
        y=dow_counts['trips'],
        marker_color=dow_counts['color'],
        marker_line_color='black',
        marker_line_width=0.5,
        hovertemplate='%{x}<br>Trips: %{y:,.0f}<extra></extra>'
    )
])

fig3.update_layout(
    title="Total Trips per Day of Week",
    xaxis_title="Day of Week",
    yaxis_title="Total Trips",
    height=500,
    showlegend=False
)

st.plotly_chart(fig3, use_container_width=True)

st.markdown("""
Perhaps unexpectedly, we **don't see a clear divide between weekends & weekdays in terms of trip volume**- while Sunday is the day with the least trips, Saturday performs better than Monday.
""")
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Chart 4 — Weather impact (PLOTLY, smoothed only)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Chart 4: Weather's Impact on Ridership")
st.markdown("*Daily bike trips and temperature (7-day smoothed)*")

@st.cache_data(show_spinner=True)
def load_daily_aggregates() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR/"daily_aggregates.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['rides_7'] = df['bike_rides_daily'].rolling(7, min_periods=1).mean()
    df['temp_7'] = df['avgTemp'].rolling(7, min_periods=1).mean()
    return df.reset_index()

df_daily = load_daily_aggregates()

fig4 = go.Figure()

# Bike rides (left y-axis) - WEEKEND COLOR
fig4.add_trace(go.Scatter(
    x=df_daily['date'],
    y=df_daily['rides_7'],
    mode='lines',
    name='Bike rides (7-day MA)',
    line=dict(color=WEEKEND_COLOR, width=2),
    yaxis='y1',
    hovertemplate='%{x|%b %d}<br>Rides: %{y:,.0f}<extra></extra>'
))

# Temperature (right y-axis) - WEEKDAY COLOR
fig4.add_trace(go.Scatter(
    x=df_daily['date'],
    y=df_daily['temp_7'],
    mode='lines',
    name='Avg temp (7-day MA)',
    line=dict(color=WEEKDAY_COLOR, width=2),
    yaxis='y2',
    hovertemplate='%{x|%b %d}<br>Temp: %{y:.1f}°C<extra></extra>'
))

fig4.update_layout(
    title='Bike Trips vs Temperature – 2022 (smoothed)',
    xaxis=dict(title='Date'),
    yaxis=dict(
        title='Bike rides',
        title_font=dict(color=WEEKEND_COLOR),
        tickfont=dict(color=WEEKEND_COLOR)
    ),
    yaxis2=dict(
        title='Temperature (°C)',
        title_font=dict(color=WEEKDAY_COLOR),
        tickfont=dict(color=WEEKDAY_COLOR),
        overlaying='y',
        side='right'
    ),
    hovermode='x unified',
    height=500
)

st.plotly_chart(fig4, use_container_width=True)

st.markdown("""
**Even though most trips are under 10 minutes, weather is a factor in trip volume**: warm days pull ridership up; winter cools the system down—useful for planning fleet scale.
""")
st.markdown("---")