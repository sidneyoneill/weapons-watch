# backend/app.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Literal
import geopandas as gpd
import pandas as pd
import json
import uvicorn
from enum import Enum
import os
# from prepare_arms_trade_matrix import prepare_arms_trade_matrix

# Define DataMode enum for type safety
class DataMode(str, Enum):
    TOTAL = "total"
    GDP = "gdp"

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware with development configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
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
    },
    "trade": {
        "tidy": None,
        "processed": None
    }
}

# Base path for data files
BASE_PATH = "../data"

print(f"Using data path: {BASE_PATH}")

# Function to load data based on mode
def load_data(mode: DataMode = DataMode.TOTAL):
    """Load data based on the specified mode (total or gdp)"""
    if data_cache[mode]["json"] is None:
        try:
            # Load JSON data - use BASE_PATH for all files
            if mode == DataMode.TOTAL:
                json_file = os.path.join(BASE_PATH, 'sipri_milex_data_nested.json')
            else:
                json_file = os.path.join(BASE_PATH, 'sipri_milex_gdp_data_nested.json')
                
            try:
                with open(json_file, 'r') as f:
                    data_cache[mode]["json"] = json.load(f)
            except FileNotFoundError:
                # If GDP nested JSON doesn't exist, we'll need to use the tidy CSV instead
                if mode == DataMode.GDP:
                    tidy_file = os.path.join(BASE_PATH, 'sipri_milex_gdp_data_tidy.csv')
                    if os.path.exists(tidy_file):
                        data_cache[mode]["tidy"] = pd.read_csv(tidy_file)
                    else:
                        print(f"Warning: Could not find GDP data file at {tidy_file}")
                else:
                    raise
        except Exception as e:
            raise RuntimeError(f"Error loading JSON dataset for mode {mode}: {e}")
        
        try:
            # Load GeoJSON data
            if mode == DataMode.TOTAL:
                geo_file = os.path.join(BASE_PATH, 'sipri_milex_data_merged.geojson')
            else:
                geo_file = os.path.join(BASE_PATH, 'sipri_milex_gdp_data_merged.geojson')
                
            data_cache[mode]["geo"] = gpd.read_file(geo_file)
        except Exception as e:
            raise RuntimeError(f"Error loading geo dataset for mode {mode}: {e}")
    
    return data_cache[mode]

# Function to load trade data
def load_trade_data():
    """Load the SIPRI trade data from CSV"""
    if data_cache["trade"]["tidy"] is None or data_cache["trade"]["processed"] is None:
        try:
            # Use absolute paths
            tidy_file = os.path.join(BASE_PATH, 'sipri_trade_data_tidy.csv')
            processed_file = os.path.join(BASE_PATH, 'sipri_trade_data_processed.csv')
            
            if os.path.exists(tidy_file):
                data_cache["trade"]["tidy"] = pd.read_csv(tidy_file)
                print(f"Loaded tidy trade data from {tidy_file}")
                print(f"Column names: {data_cache['trade']['tidy'].columns.tolist()}")
            else:
                print(f"Tidy file not found at {tidy_file}")
                
            if os.path.exists(processed_file):
                data_cache["trade"]["processed"] = pd.read_csv(processed_file)
                print(f"Loaded processed trade data from {processed_file}")
                print(f"Column names: {data_cache['trade']['processed'].columns.tolist()}")
            else:
                print(f"Processed file not found at {processed_file}")
            
            if data_cache["trade"]["tidy"] is None and data_cache["trade"]["processed"] is None:
                raise FileNotFoundError("Could not find trade data files")
                
        except Exception as e:
            raise RuntimeError(f"Error loading trade dataset: {e}")
    
    return data_cache["trade"]

# Load default data at startup
try:
    print("Loading 'total' data...")
    load_data(DataMode.TOTAL)
    print("Successfully loaded 'total' data")
except Exception as e:
    print(f"Error loading 'total' data at startup: {e}")
    print("The application will continue, but some endpoints may not work properly")
    
try:
    print("Loading 'gdp' data...")
    load_data(DataMode.GDP)
    print("Successfully loaded 'gdp' data")
except Exception as e:
    print(f"Error loading 'gdp' data at startup: {e}")
    print("The application will continue, but GDP-related endpoints may not work properly")

try:
    print("Loading trade data...")
    load_trade_data()
    print("Successfully loaded trade data")
except Exception as e:
    print(f"Error loading trade data at startup: {e}")
    print("The application will continue, but trade-related endpoints may not work properly")

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

@app.get("/trade_data", response_model=Dict[str, Any])
def get_trade_data(
    start_year: int = Query(None, description="Filter data from this year onwards"),
    end_year: int = Query(None, description="Filter data until this year"),
    recipient: str = Query(None, description="Filter by recipient country"),
    supplier: str = Query(None, description="Filter by supplier country")
) -> Dict[str, Any]:
    """
    Returns arms trade data with optional filtering by year range and countries.
    
    Parameters:
    - start_year: Optional starting year filter
    - end_year: Optional ending year filter
    - recipient: Optional recipient country filter
    - supplier: Optional supplier country filter
    """
    data = load_trade_data()
    df = data["tidy"]
    
    # Apply filters if provided
    if start_year is not None:
        df = df[df["Year"] >= start_year]
    if end_year is not None:
        df = df[df["Year"] <= end_year]
    if recipient is not None:
        df = df[df["Recipient"].str.lower() == recipient.lower()]
    if supplier is not None:
        df = df[df["Supplier"].str.lower() == supplier.lower()]
    
    # Convert filtered data to list of records
    trades = []
    for _, row in df.iterrows():
        trades.append({
            "year": int(row["Year"]),
            "recipient": row["Recipient"],
            "recipient_iso": row["Recipient ISO"],
            "supplier": row["Supplier"],
            "supplier_iso": row["Supplier ISO"],
            "value": float(row["Value TIV"]) if "Value TIV" in row else None,
            "weapon_category": row["Weapon Category"] if "Weapon Category" in row else None,
            "weapon_description": row["Weapon Description"] if "Weapon Description" in row else None,
            "order_date": row["Order Date"] if "Order Date" in row else None,
            "delivery_date": row["Delivery Year"] if "Delivery Year" in row else None
        })
    
    return {"trades": trades}

@app.get("/trade_partners/{country}")
def get_trade_partners(country: str):
    """
    Returns trade partners for the specified country.
    
    Parameters:
    - country: The name of the country
    """
    try:
        data = load_trade_data()
        
        # Determine which dataset to use and which columns are available
        trade_data = None
        value_column = None
        supplier_column = None
        recipient_column = None
        
        # Try the processed data first
        if data["processed"] is not None:
            trade_data = data["processed"]
            # Print out column names for debugging
            print(f"Available columns in processed data: {trade_data.columns.tolist()}")
            
            # Set column names based on what's available
            if 'Supplier' in trade_data.columns:
                supplier_column = 'Supplier'
            if 'Recipient' in trade_data.columns:
                recipient_column = 'Recipient'
            
            # Find a value column - check possible names
            for col in ['SIPRI TIV of delivered weapons', 'SIPRI TIV for total order', 'Value TIV']:
                if col in trade_data.columns:
                    value_column = col
                    break
        
        # If processed data doesn't have what we need, try tidy data
        if (trade_data is None or value_column is None or 
            supplier_column is None or recipient_column is None) and data["tidy"] is not None:
            
            trade_data = data["tidy"]
            # Print out column names for debugging
            print(f"Available columns in tidy data: {trade_data.columns.tolist()}")
            
            # Set column names based on what's available
            if 'Supplier' in trade_data.columns:
                supplier_column = 'Supplier'
            if 'Recipient' in trade_data.columns:
                recipient_column = 'Recipient'
                
            # Find a value column - check possible names
            for col in ['Value TIV', 'SIPRI TIV of delivered weapons', 'SIPRI TIV for total order']:
                if col in trade_data.columns:
                    value_column = col
                    break
        
        # If we still don't have what we need, raise an error
        if trade_data is None:
            raise ValueError("No trade data available")
        if value_column is None:
            raise ValueError("No value column found in trade data")
        if supplier_column is None:
            raise ValueError("No supplier column found in trade data")
        if recipient_column is None:
            raise ValueError("No recipient column found in trade data")
        
        # Get trade relationships for the country
        country_as_supplier = trade_data[trade_data[supplier_column] == country]
        country_as_recipient = trade_data[trade_data[recipient_column] == country]
        
        # Aggregate trade volume by partner
        exports = pd.DataFrame()
        imports = pd.DataFrame()
        
        if not country_as_supplier.empty:
            exports = country_as_supplier.groupby(recipient_column).agg({
                value_column: 'sum'
            }).reset_index()
        
        if not country_as_recipient.empty:
            imports = country_as_recipient.groupby(supplier_column).agg({
                value_column: 'sum'
            }).reset_index()
        
        # Format data for response
        trade_partners = []
        
        # Add export partners
        for _, row in exports.iterrows():
            trade_partners.append({
                'country': row[recipient_column],
                'value': float(row[value_column]),
                'type': 'export'
            })
        
        # Add import partners
        for _, row in imports.iterrows():
            trade_partners.append({
                'country': row[supplier_column],
                'value': float(row[value_column]),
                'type': 'import'
            })
        
        # Sort by trade volume and get top partners
        trade_partners.sort(key=lambda x: x['value'], reverse=True)
        
        return trade_partners
    
    except Exception as e:
        print(f"Error in get_trade_partners: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing trade partners: {str(e)}"
        )

# @app.get("/arms_trade_matrix/{year}", response_model=Dict[str, Any])
# def get_arms_trade_matrix(year: int) -> Dict[str, Any]:
#     """
#     Returns the arms trade matrix for the specified year.
    
#     Parameters:
#     - year: The year to get the arms trade matrix for
#     """
#     # Check if we have a pre-generated matrix file
#     matrix_file = f'../data/arms_trade_matrices/arms_trade_matrix_{year}.json'
    
#     if os.path.exists(matrix_file):
#         # Load the pre-generated matrix
#         with open(matrix_file, 'r') as f:
#             return json.load(f)
#     else:
#         # Generate the matrix on-the-fly
#         return prepare_arms_trade_matrix(year)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
