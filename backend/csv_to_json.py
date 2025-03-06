#!/usr/bin/env python3
# backend/csv_to_json.py

import pandas as pd
import json
import os

# Define paths
input_file_absolute = 'data/sipri_milex_data_tidy.csv'
input_file_percentage = 'data/sipri_milex_gdp_data_tidy.csv'
output_file = 'data/sipri_milex_data_nested.json'

# Read the CSV files
df_absolute = pd.read_csv(input_file_absolute)
df_percentage = pd.read_csv(input_file_percentage)

# Rename columns to avoid confusion after merge
df_absolute = df_absolute.rename(columns={'Expenditure': 'Expenditure_Absolute'})
df_percentage = df_percentage.rename(columns={'Expenditure': 'Expenditure_GDP'})

# Merge the dataframes on Country, Year and ISO_Code
# Use outer join to keep data even if it's missing in one of the sources
merged_df = pd.merge(
    df_absolute, 
    df_percentage, 
    on=['Country', 'Year', 'ISO_Code'],
    how='outer'
)

# Create the nested JSON structure
countries_data = []

# Group by country
for country, group in merged_df.groupby('Country'):
    # Sort by year
    group = group.sort_values('Year')
    
    # Get ISO code
    iso_code = group['ISO_Code'].iloc[0]
    
    # Create time series for this country
    time_series = []
    for _, row in group.iterrows():
        # Handle potentially missing data with None
        absolute_value = float(row['Expenditure_Absolute']) if pd.notna(row.get('Expenditure_Absolute')) else None
        gdp_percentage = float(row['Expenditure_GDP']) if pd.notna(row.get('Expenditure_GDP')) else None
        
        time_series.append({
            "year": int(row['Year']),
            "military_expenditure": absolute_value,
            "military_expenditure_gdp": gdp_percentage,
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
print(f"Added both absolute military expenditure and percentage of GDP data")