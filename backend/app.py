# backend/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import geopandas as gpd
import json
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load JSON data and geographic data (GeoJSON) at startup.
try:
    with open('../data/sipri_milex_data_nested.json', 'r') as f:
        json_data = json.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading JSON dataset: {e}")

try:
    geo_gdf = gpd.read_file('../data/sipri_milex_data_merged.geojson')
except Exception as e:
    raise RuntimeError(f"Error loading geo dataset: {e}")

@app.get("/countries", response_model=Dict[str, List[str]])
def get_countries() -> Dict[str, List[str]]:
    """
    Returns a sorted list of unique countries that have military expenditure data.
    """
    countries = sorted([country["name"] for country in json_data["countries"]])
    return {"countries": countries}

@app.get("/expenditure/{country}", response_model=Dict[str, Any])
def get_expenditure(country: str) -> Dict[str, Any]:
    """
    Returns the time series of military expenditure for the specified country.
    The matching is case-insensitive.
    """
    # Find the country in the JSON data (case-insensitive matching)
    country_data = next((c for c in json_data["countries"] 
                       if c["name"].lower() == country.lower()), None)
    
    if not country_data:
        raise HTTPException(status_code=404, detail="Country not found")
    
    # The time series is already in the right format in our JSON
    return {
        "country": country_data["name"],
        "time_series": country_data["time_series"]
    }

@app.get("/geo_data", response_model=Dict[str, Any])
def get_static_geo_data() -> Dict[str, Any]:
    """
    Returns the static geographic data (country boundaries) from the GeoJSON file.
    This endpoint does not filter by year.
    """
    # Optionally, if your GeoJSON file already only has static boundaries, you can simply return it.
    static_geo_json = geo_gdf.to_json()
    return {"data": static_geo_json}

@app.get("/all_expenditures", response_model=Dict[str, Any])
def get_all_expenditures() -> Dict[str, Any]:
    """
    Returns the time series of military expenditure for all countries.
    """
    all_time_series = []
    
    for country in json_data["countries"]:
        country_name = country["name"]
        for entry in country["time_series"]:
            all_time_series.append({
                "Country": country_name,
                "Year": entry["year"],
                "Expenditure": entry["military_expenditure"]
            })
    
    # Sort by year
    all_time_series.sort(key=lambda x: x["Year"])
    
    return {"time_series": all_time_series}

@app.get("/country/{iso_code}", response_model=Dict[str, Any])
def get_country_by_iso(iso_code: str) -> Dict[str, Any]:
    """
    Returns country data by ISO code.
    The matching is case-insensitive.
    """
    # Find the country in the JSON data by ISO code (case-insensitive matching)
    country_data = next((c for c in json_data["countries"] 
                       if c["ISO"].lower() == iso_code.lower()), None)
    
    if not country_data:
        raise HTTPException(status_code=404, detail="Country not found")
    
    return country_data

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
