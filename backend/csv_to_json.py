#!/usr/bin/env python3
# backend/csv_to_json.py

import pandas as pd
import json
import os

# Define paths
input_file = '../data/sipri_milex_data_tidy.csv'
output_file = '../data/sipri_milex_data_nested.json'

# Read the CSV file
df = pd.read_csv(input_file)

# Create the nested JSON structure
countries_data = []

# Group by country
for country, group in df.groupby('Country'):
    # Sort by year
    group = group.sort_values('Year')
    
    # Get ISO code
    iso_code = group['ISO_Code'].iloc[0]
    
    # Create time series for this country
    time_series = []
    for _, row in group.iterrows():
        time_series.append({
            "year": int(row['Year']),
            "military_expenditure": float(row['Expenditure']),
            "arms_trading": {
                "imports": [],  # Empty placeholder for future arms import data
                "exports": []   # Empty placeholder for future arms export data
            }
        })
    
    # Add country data to the list
    countries_data.append({
        "name": country,
        "ISO": iso_code,
        "time_series": time_series
    })

# Create the final JSON structure
json_data = {
    "countries": countries_data
}

# Save to JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"Conversion complete. JSON data saved to {output_file}") 