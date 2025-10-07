# NYC CitiBike Distribution Analysis 2022

**An interactive data analytics project exploring user behavior patterns and distribution challenges in New York City's bike-sharing system**

---

## Overview

In this project, I take on the role of lead analyst for CitiBike, New York City's bike-sharing company. The analysis examines 2022 trip data to help the business strategy team assess the current logistics model and identify opportunities to improve bike distribution across the city.

## Objective

The goal is to conduct a descriptive analysis of existing data and discover actionable insights that will:
- Improve fleet operations
- Identify expansion opportunities
- Help circumvent bike availability issues


## Context

CitiBike's popularity has surged since its 2013 launch, with the COVID-19 pandemic accelerating adoption as New Yorkers sought sustainable, socially-distanced transportation options. This increased demand has created distribution challenges:
- Popular stations running out of bikes
- Stations becoming too full to dock returned bikes
- Rising customer complaints

## Data & Tools

**Data Sources:**
- CitiBike open-source trip data for 2022 (https://s3.amazonaws.com/tripdata/index.html)
- Weather data from NOAA's API service(https://www.noaa.gov/)

**Technologies:**
- **Python Libraries:** pandas, Matplotlib, Seaborn, Plotly, Kepler.gl
- **Dashboard:** Streamlit
- **Analysis:** Jupyter Notebooks

## Data Access

Due to GitHub's file size limits, the two largest datasets are hosted on Google Drive:

**[📥 Download the data files here](https://drive.google.com/drive/folders/18dn7QjYPa3z1ZIkUEGr9zHzWnwQxSUMz)**

The download includes:
- `nyc_2022_essential_data.csv` - Complete 2022 trip data
- `station_to_nta.csv` - Station-to-neighborhood mapping

These files are required for running the analysis notebooks. The dashboard uses lighter, aggregated versions of this data (found in the "Prepared Data" folder).

Note: Each page of the dashboard indicates which preparation notebooks were used to create the underlying data.

About This Project
This analysis was completed as the capstone project for CareerFoundry's Data Analyst Programme. The project demonstrates end-to-end data analysis skills including data cleaning, exploratory analysis, geospatial visualization, and interactive dashboard development.


