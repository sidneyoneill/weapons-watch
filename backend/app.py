# backend/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import geopandas as gpd
import pandas as pd
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

# Load attribute data (CSV) and geographic data (GeoJSON) at startup.
try:
    attribute_df = pd.read_csv('../data/sipri_milex_data_tidy.csv')
except Exception as e:
    raise RuntimeError(f"Error loading attribute dataset: {e}")

try:
    geo_gdf = gpd.read_file('../data/sipri_milex_data_merged.geojson')
except Exception as e:
    raise RuntimeError(f"Error loading geo dataset: {e}")

@app.get("/countries", response_model=Dict[str, List[str]])
def get_countries() -> Dict[str, List[str]]:
    """
    Returns a sorted list of unique countries that have military expenditure data.
    """
    countries = sorted(attribute_df['Country'].unique().tolist())
    return {"countries": countries}

@app.get("/expenditure/{country}", response_model=Dict[str, Any])
def get_expenditure(country: str) -> Dict[str, Any]:
    """
    Returns the time series of military expenditure for the specified country.
    The matching is case-insensitive.
    """
    # Filter the attribute data for the selected country (case-insensitive matching)
    filtered = attribute_df[attribute_df['Country'].str.lower() == country.lower()]
    if filtered.empty:
        raise HTTPException(status_code=404, detail="Country not found")
    
    # Sort by Year and convert to a list of records
    time_series = filtered[['Year', 'Expenditure']].sort_values(by='Year').to_dict(orient='records')
    return {"country": country, "time_series": time_series}

@app.get("/geo_data", response_model=Dict[str, Any])
def get_static_geo_data() -> Dict[str, Any]:
    """
    Returns the static geographic data (country boundaries) from the GeoJSON file.
    This endpoint does not filter by year.
    """
    # Optionally, if your GeoJSON file already only has static boundaries, you can simply return it.
    static_geo_json = geo_gdf.to_json()
    return {"data": static_geo_json}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
