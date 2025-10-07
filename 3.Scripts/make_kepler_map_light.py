from _paths import csv_path
import pandas as pd
from keplergl import KeplerGl
from pathlib import Path

# Load
stations = pd.read_csv(csv_path("citibike_2022_stations_high_flow.csv"))
flows    = pd.read_csv(csv_path("citibike_2022_flows_1500plus.csv"))

# === SLIM DOWN THE DATA ===
# keep only needed columns
stations = stations[["lat","lng","total_trips"]].copy()
flows    = flows[["start_lat","start_lng","end_lat","end_lng","trips"]].copy()

# round coords to 5 decimals to shrink JSON
for col in ["lat","lng"]:
    stations[col] = stations[col].round(5)
for col in ["start_lat","start_lng","end_lat","end_lng"]:
    flows[col] = flows[col].round(5)

# keep top-N flows by trips to control size (tune N if needed)
N = 400
flows = flows.sort_values("trips", ascending=False).head(N).reset_index(drop=True)

# center
center_lat = stations["lat"].mean()
center_lon = stations["lng"].mean()

flare_like = {
    "name": "flare_like",
    "type": "sequential",
    "category": "Uber",
    "colors": ["#2D1E3E", "#6B1F73", "#A22C7E", "#D6456C", "#F77C48", "#FDBD3C"]
}

cfg = {
    "version":"v1",
    "config":{
        "visState":{
            "filters":[{
                "dataId":"Flows","id":"trips_filter","name":["trips"],
                "type":"range","value":[int(flows["trips"].min()), int(flows["trips"].max())],
                "enlarged": True
            }],
            "layers":[
                {"id":"stations-point","type":"point",
                 "config":{"dataId":"Stations","label":"Stations",
                           "columns":{"lat":"lat","lng":"lng"},
                           "isVisible":True,"visConfig":{"radius":4,"colorRange":flare_like}},
                 "visualChannels":{
                     "colorField":{"name":"total_trips","type":"integer"},
                     "colorScale":"quantile",
                     "sizeField":{"name":"total_trips","type":"integer"},
                     "sizeScale":"sqrt"}},
                {"id":"flows-arc","type":"arc",
                 "config":{"dataId":"Flows","label":"OD Flows",
                           "columns":{"lat0":"start_lat","lng0":"start_lng",
                                      "lat1":"end_lat","lng1":"end_lng"},
                           "isVisible":True,"visConfig":{"thickness":4,"opacity":0.7,"colorRange":flare_like}},
                 "visualChannels":{
                     "sizeField":{"name":"trips","type":"integer"},
                     "sizeScale":"sqrt",
                     "colorField":{"name":"trips","type":"integer"},
                     "colorScale":"quantile"}}
            ]},
            "mapState":{"latitude":float(center_lat),"longitude":float(center_lon),"zoom":12}
    }
}

m = KeplerGl(height=650, config=cfg)
m.add_data(stations, "Stations")
m.add_data(flows,    "Flows")

out = Path("3.Scripts/dashboard/assets")
out.mkdir(parents=True, exist_ok=True)
outfile = out / "kepler_map_light.html"
m.save_to_html(file_name=str(outfile), read_only=True)
print("Saved:", outfile)
