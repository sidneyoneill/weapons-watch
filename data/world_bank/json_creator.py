import csv
import json
import os
import datetime
from collections import defaultdict

def import_and_clean_csv(csv_path):
    """
    Import and clean the World Bank CSV data.
    
    Args:
        csv_path: Path to the World Bank CSV file
    
    Returns:
        cleaned_data: A structured dictionary of the cleaned data
        indicators_metadata: Dictionary with indicator metadata
        years: List of years from the CSV headers
    """
    print(f"Reading and cleaning CSV from: {csv_path}")
    
    # Dictionary to store the cleaned data and indicators metadata
    cleaned_data = defaultdict(lambda: defaultdict(dict))
    indicators_metadata = {}
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        # Extract headers with years
        headers = next(reader)
        country_name_idx = headers.index("Country Name")
        country_code_idx = headers.index("Country Code")
        indicator_name_idx = headers.index("Indicator Name")
        indicator_code_idx = headers.index("Indicator Code")
        
        # Extract years from headers (starting from index 4)
        years = headers[4:]
        
        # Process each row
        for row in reader:
            if len(row) < 4:
                continue
            
            country_name = row[country_name_idx]
            country_code = row[country_code_idx]
            indicator_name = row[indicator_name_idx]
            indicator_code = row[indicator_code_idx]
            
            # Store indicator metadata if not already stored
            if indicator_code not in indicators_metadata:
                indicators_metadata[indicator_code] = {
                    "name": indicator_name
                }
            
            # Get values for each year
            values = {}
            for i, year in enumerate(years):
                if i < len(row) - 4:
                    value = row[i + 4].strip()
                    if value:  # Only store non-empty values
                        try:
                            # Try to convert to float/int
                            numeric_value = float(value)
                            if numeric_value.is_integer():
                                values[year] = int(numeric_value)
                            else:
                                values[year] = numeric_value
                        except ValueError:
                            # Store as string if can't convert to number
                            values[year] = value
            
            # Store in the cleaned data structure
            if values:  # Only store if there are values
                cleaned_data[country_code][indicator_code] = values
    
    # Convert defaultdict to regular dict for easier JSON serialization
    return {k: dict(v) for k, v in cleaned_data.items()}, indicators_metadata, years

def convert_to_json(cleaned_data, indicators_metadata, countries_metadata, output_path):
    """
    Convert cleaned data to JSON in a normalized structure.
    
    Args:
        cleaned_data: Cleaned data from import_and_clean_csv
        indicators_metadata: Dictionary with indicator metadata
        countries_metadata: Dictionary mapping country codes to country names
        output_path: Path to save the output JSON
    """
    # Create the countries list
    countries_list = []
    
    for country_code, indicators_data in cleaned_data.items():
        country_name = countries_metadata.get(country_code, country_code)
        
        # Create country object with normalized structure
        country_obj = {
            "name": country_name,
            "ISO": country_code,
            "data": indicators_data
        }
        
        countries_list.append(country_obj)
    
    # Get list of included indicators for metadata
    indicators_included = [indicators_metadata[code]["name"] for code in indicators_metadata.keys()]
    
    # Create the final output structure
    output_data = {
        "metadata": {
            "source": "World Bank",
            "description": "World Bank Development Indicators",
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d"),
            "indicators_included": indicators_included[:10] + (["..."] if len(indicators_included) > 10 else [])
        },
        "indicators": indicators_metadata,
        "countries": countries_list
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(output_data, jsonfile, indent=2)
    
    print(f"Successfully converted CSV to JSON. Output saved to: {output_path}")
    print(f"Processed {len(countries_list)} countries with {len(indicators_metadata)} indicators")

if __name__ == "__main__":
    csv_path = 'data/world_bank/WDICSV_modified.csv'
    output_path = 'data/world_bank/world_bank_data_normalized.json'
    
    # Import and clean the CSV
    cleaned_data, indicators_metadata, years = import_and_clean_csv(csv_path)
    
    # Create a dictionary mapping country codes to country names
    countries_metadata = {}
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:
            if len(row) >= 2:
                countries_metadata[row[1]] = row[0]
    
    # Convert to JSON
    convert_to_json(cleaned_data, indicators_metadata, countries_metadata, output_path)