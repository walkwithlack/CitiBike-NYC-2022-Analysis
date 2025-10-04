"""
Pre-aggregate CitiBike data for efficient dashboard loading.
Run this ONCE to create lightweight data files for the dashboard.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(r"C:\Users\magia\OneDrive\Desktop\NY_Citi_Bike\2.Data\Prepared Data")
INPUT_FILE = DATA_DIR / "nyc_2022_essential_data.csv"
OUTPUT_DIR = DATA_DIR / "aggregated"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading main dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

# Parse datetimes
df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
df = df.dropna(subset=['started_at', 'ended_at'])

print(f"Loaded {len(df):,} trips")

# ==================== Chart 1: Trip Duration Sample ====================
print("\n1. Creating trip duration sample...")
df['tripduration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

# Clean and filter
df_duration = df[
    (df['ended_at'] >= df['started_at']) &
    (df['tripduration_min'] >= 1) &
    (df['tripduration_min'] <= 120) &
    (df['tripduration_min'] <= 75)
].copy()

# Sample 100k trips for histogram (more than enough for visualization)
duration_sample = df_duration['tripduration_min'].sample(n=min(100000, len(df_duration)), random_state=42)
duration_sample.to_csv(OUTPUT_DIR / "trip_durations.csv", index=False, header=['tripduration_min'])
print(f"   Saved {len(duration_sample):,} duration samples")

# ==================== Chart 2: Weekday vs Weekend Patterns ====================
print("\n2. Creating weekday/weekend hourly patterns...")
df['hour'] = df['started_at'].dt.hour
df['date'] = df['started_at'].dt.date
weekend_days = {"Saturday", "Sunday"}
df['period'] = df['started_at'].dt.day_name().isin(weekend_days).map({True: "Weekend", False: "Weekday"})

overlay = (df
           .groupby(["period", "date", "hour"])
           .size()
           .groupby(["period", "hour"])
           .mean()
           .reset_index(name="trips_per_day"))

overlay.to_csv(OUTPUT_DIR / "hourly_patterns.csv", index=False)
print(f"   Saved {len(overlay)} hourly pattern records")

# ==================== Chart 3: Day of Week Totals ====================
print("\n3. Creating day of week totals...")
dow_counts = (df.groupby(df['started_at'].dt.day_name())
                .size()
                .reset_index(name='trips'))
dow_counts.columns = ['day_name', 'trips']
dow_counts.to_csv(OUTPUT_DIR / "day_of_week_totals.csv", index=False)
print(f"   Saved {len(dow_counts)} day-of-week records")

# ==================== Chart 4: Daily Trips and Temperature ====================
print("\n4. Creating daily aggregates (trips + temperature)...")
df_daily = df.groupby('date').agg(
    bike_rides_daily=('date', 'size'),
    avgTemp=('avgTemp', 'mean')
).reset_index()

df_daily.to_csv(OUTPUT_DIR / "daily_aggregates.csv", index=False)
print(f"   Saved {len(df_daily)} daily records")

# ==================== Chart 6: Station and Flow Data ====================
print("\n5. Creating station and flow data...")

# Check if pre-computed files exist
stations_file = DATA_DIR / "citibike_2022_stations_filtered_90_percent.csv"
flows_file = DATA_DIR / "citibike_2022_od_counts_90_percent.csv"

if stations_file.exists() and flows_file.exists():
    print("   Loading existing station and flow files...")
    stations = pd.read_csv(stations_file)
    flows = pd.read_csv(flows_file)
    
    # Filter flows to 1500+ trips for visualization
    flows_filtered = flows[flows['trip_count'] >= 1500]
    
    # Save to aggregated folder
    stations.to_csv(OUTPUT_DIR / "stations.csv", index=False)
    flows_filtered.to_csv(OUTPUT_DIR / "flows_major.csv", index=False)
    print(f"   Saved {len(stations):,} stations and {len(flows_filtered):,} major flows")
else:
    print("   Computing station and flow data from scratch...")
    
    # Get top 90% of stations by volume
    start_counts = df['start_station_name'].value_counts()
    end_counts = df['end_station_name'].value_counts()
    total_trips = len(df)
    
    cumsum_start = start_counts.cumsum() / total_trips
    cumsum_end = end_counts.cumsum() / total_trips
    
    stations_for_90_start = (cumsum_start <= 0.9).sum()
    stations_for_90_end = (cumsum_end <= 0.9).sum()
    
    top_90_start = start_counts.head(stations_for_90_start).index.tolist()
    top_90_end = end_counts.head(stations_for_90_end).index.tolist()
    
    df_filtered = df[
        (df['start_station_name'].isin(top_90_start)) & 
        (df['end_station_name'].isin(top_90_end))
    ]
    
    print(f"   Filtered to {len(df_filtered):,} trips ({len(df_filtered)/len(df)*100:.1f}%)")
    
    # Create stations data
    stations = (
        df_filtered.groupby(['start_station_name', 'start_lat', 'start_lng'])
        .size()
        .reset_index(name='starts')
        .rename(columns={'start_station_name': 'station', 'start_lat': 'lat', 'start_lng': 'lng'})
    )
    
    ends_data = (
        df_filtered.groupby(['end_station_name', 'end_lat', 'end_lng'])
        .size()
        .reset_index(name='ends') 
        .rename(columns={'end_station_name': 'station', 'end_lat': 'lat', 'end_lng': 'lng'})
    )
    
    stations = stations.merge(ends_data, on=['station', 'lat', 'lng'], how='outer').fillna(0)
    stations['total_trips'] = (stations['starts'] + stations['ends']).astype('int64')
    stations[['starts', 'ends']] = stations[['starts', 'ends']].astype('int64')
    
    # Create OD flows (process in chunks to avoid memory issues)
    print("   Computing OD flows in chunks...")
    chunk_size = 5_000_000
    od_list = []
    
    for i in range(0, len(df_filtered), chunk_size):
        chunk = df_filtered.iloc[i:i+chunk_size]
        od_chunk = (
            chunk
            .groupby(['start_station_name','end_station_name'])
            .size()
        )
        od_list.append(od_chunk)
        print(f"      Processed chunk {i//chunk_size + 1}")
    
    od_series = pd.concat(od_list).groupby(level=[0,1]).sum()
    flows = od_series.reset_index(name="trip_count")
    
    # Add coordinates
    start_coords = df_filtered[['start_station_name', 'start_lat', 'start_lng']].drop_duplicates('start_station_name')
    end_coords = df_filtered[['end_station_name', 'end_lat', 'end_lng']].drop_duplicates('end_station_name')
    
    flows = flows.merge(start_coords, on='start_station_name', how='left')
    flows = flows.merge(end_coords, on='end_station_name', how='left')
    
    # Filter to major flows only (1500+)
    flows_filtered = flows[flows['trip_count'] >= 1500]
    
    # Save
    stations.to_csv(OUTPUT_DIR / "stations.csv", index=False)
    flows_filtered.to_csv(OUTPUT_DIR / "flows_major.csv", index=False)
    print(f"   Saved {len(stations):,} stations and {len(flows_filtered):,} major flows")

print("\n" + "="*60)
print("PRE-PROCESSING COMPLETE!")
print("="*60)
print(f"\nAggregated files saved to: {OUTPUT_DIR}")
print("\nFiles created:")
print("  - trip_durations.csv (100k samples)")
print("  - hourly_patterns.csv (weekday/weekend by hour)")
print("  - day_of_week_totals.csv (7 records)")
print("  - daily_aggregates.csv (365 records)")
print("  - stations.csv (station locations + volumes)")
print("  - flows_major.csv (major OD flows, 1500+ trips)")
print("\nYou can now run the dashboard with these lightweight files!")