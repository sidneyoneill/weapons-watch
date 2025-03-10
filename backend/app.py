# backend/app.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Literal
import geopandas as gpd
import pandas as pd
import json
import uvicorn
from enum import Enum

# Define DataMode enum for type safety
class DataMode(str, Enum):
    TOTAL = "total"
    GDP = "gdp"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data cache to store loaded data
data_cache = {
    "total": {
        "json": None,
        "geo": None,
        "tidy": None
    },
    "gdp": {
        "json": None,
        "geo": None,
        "tidy": None
    }
}

# Function to load data based on mode
def load_data(mode: DataMode = DataMode.TOTAL):
    """Load data based on the specified mode (total or gdp)"""
    if data_cache[mode]["json"] is None:
        try:
            # Load JSON data
            json_file = f'../data/sipri_milex_data_nested.json' if mode == DataMode.TOTAL else f'../data/sipri_milex_gdp_data_nested.json'
            try:
                with open(json_file, 'r') as f:
                    data_cache[mode]["json"] = json.load(f)
            except FileNotFoundError:
                # If GDP nested JSON doesn't exist, we'll need to use the tidy CSV instead
                if mode == DataMode.GDP:
                    tidy_file = f'../data/sipri_milex_gdp_data_tidy.csv'
                    data_cache[mode]["tidy"] = pd.read_csv(tidy_file)
                else:
                    raise
        except Exception as e:
            raise RuntimeError(f"Error loading JSON dataset for mode {mode}: {e}")
        
        try:
            # Load GeoJSON data
            geo_file = f'../data/sipri_milex_data_merged.geojson' if mode == DataMode.TOTAL else f'../data/sipri_milex_gdp_data_merged.geojson'
            data_cache[mode]["geo"] = gpd.read_file(geo_file)
        except Exception as e:
            raise RuntimeError(f"Error loading geo dataset for mode {mode}: {e}")
    
    return data_cache[mode]

# Load default data at startup
load_data(DataMode.TOTAL)

@app.get("/countries", response_model=Dict[str, List[str]])
def get_countries() -> Dict[str, List[str]]:
    """
    Returns a sorted list of unique countries that have military expenditure data.
    """
    data = load_data(DataMode.TOTAL)
    countries = sorted([country["name"] for country in data["json"]["countries"]])
    return {"countries": countries}

@app.get("/expenditure/{country}", response_model=Dict[str, Any])
def get_expenditure(
    country: str, 
    mode: DataMode = Query(DataMode.TOTAL, description="Data mode: total or gdp")
) -> Dict[str, Any]:
    """
    Returns the time series of military expenditure for the specified country.
    The matching is case-insensitive.
    
    Parameters:
    - country: The name of the country
    - mode: 'total' for absolute military expenditure, 'gdp' for percentage of GDP
    """
    data = load_data(mode)
    
    # If we have JSON data, use it
    if data["json"]:
        # Find the country in the JSON data (case-insensitive matching)
        country_data = next((c for c in data["json"]["countries"] 
                           if c["name"].lower() == country.lower()), None)
        
        if not country_data:
            raise HTTPException(status_code=404, detail="Country not found")
        
        # The time series is already in the right format in our JSON
        return {
            "country": country_data["name"],
            "time_series": country_data["time_series"]
        }
    # Otherwise use the tidy CSV data
    elif data["tidy"] is not None:
        # Filter the dataframe for the specified country
        country_df = data["tidy"][data["tidy"]["Country"].str.lower() == country.lower()]
        
        if country_df.empty:
            raise HTTPException(status_code=404, detail="Country not found")
        
        # Convert to the expected format
        country_name = country_df["Country"].iloc[0]
        time_series = []
        
        for _, row in country_df.iterrows():
            time_series.append({
                "year": int(row["Year"]),
                "military_expenditure": float(row["Expenditure"])
            })
        
        return {
            "country": country_name,
            "time_series": time_series
        }
    else:
        raise HTTPException(status_code=500, detail="Data not available")

@app.get("/geo_data", response_model=Dict[str, Any])
def get_static_geo_data(
    mode: DataMode = Query(DataMode.TOTAL, description="Data mode: total or gdp")
) -> Dict[str, Any]:
    """
    Returns the static geographic data (country boundaries) from the GeoJSON file.
    
    Parameters:
    - mode: 'total' for absolute military expenditure, 'gdp' for percentage of GDP
    """
    data = load_data(mode)
    static_geo_json = data["geo"].to_json()
    return {"data": static_geo_json}

@app.get("/all_expenditures", response_model=Dict[str, Any])
def get_all_expenditures(
    mode: DataMode = Query(DataMode.TOTAL, description="Data mode: total or gdp")
) -> Dict[str, Any]:
    """
    Returns the time series of military expenditure for all countries.
    
    Parameters:
    - mode: 'total' for absolute military expenditure, 'gdp' for percentage of GDP
    """
    data = load_data(mode)
    all_time_series = []
    
    # If we have JSON data, use it
    if data["json"]:
        for country in data["json"]["countries"]:
            country_name = country["name"]
            for entry in country["time_series"]:
                all_time_series.append({
                    "Country": country_name,
                    "Year": entry["year"],
                    "Expenditure": entry["military_expenditure"]
                })
    # Otherwise use the tidy CSV data
    elif data["tidy"] is not None:
        for _, row in data["tidy"].iterrows():
            all_time_series.append({
                "Country": row["Country"],
                "Year": int(row["Year"]),
                "Expenditure": float(row["Expenditure"])
            })
    else:
        raise HTTPException(status_code=500, detail="Data not available")
    
    # Sort by year
    all_time_series.sort(key=lambda x: x["Year"])
    
    return {"time_series": all_time_series}

@app.get("/country/{iso_code}", response_model=Dict[str, Any])
def get_country_by_iso(
    iso_code: str,
    mode: DataMode = Query(DataMode.TOTAL, description="Data mode: total or gdp")
) -> Dict[str, Any]:
    """
    Returns country data by ISO code.
    The matching is case-insensitive.
    
    Parameters:
    - iso_code: The ISO code of the country
    - mode: 'total' for absolute military expenditure, 'gdp' for percentage of GDP
    """
    data = load_data(mode)
    
    if data["json"]:
        # Find the country in the JSON data by ISO code (case-insensitive matching)
        country_data = next((c for c in data["json"]["countries"] 
                           if c["ISO"].lower() == iso_code.lower()), None)
        
        if not country_data:
            raise HTTPException(status_code=404, detail="Country not found")
        
        return country_data
    else:
        raise HTTPException(status_code=500, detail="Data not available in this format")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
