#!/usr/bin/env python3

import pandas as pd
import math
import pycountry

def get_iso_code(country_name):
    """Convert country name to ISO 3166-1 alpha-3 code."""
    try:
        # Handle special cases
        if country_name == "United States":
            return "USA"
        elif country_name == "Russia":
            return "RUS"
        elif country_name == "Soviet Union":
            return "RUS"  # Map Soviet Union to Russia
        
        # Try to find the country
        country = pycountry.countries.search_fuzzy(country_name)[0]
        return country.alpha_3
    except:
        return "Unknown"

def process_trade_data():
    # Read the CSV with encoding specified
    df = pd.read_csv('data/sipri_trade_data.csv', encoding='latin-1')

    # Clean up the data
    df['Year of order'] = pd.to_numeric(df['Year of order'], errors='coerce')
    df['Number ordered'] = pd.to_numeric(df['Number ordered'], errors='coerce')
    df['Number delivered'] = pd.to_numeric(df['Number delivered'], errors='coerce')
    df['SIPRI TIV per unit'] = pd.to_numeric(df['SIPRI TIV per unit'], errors='coerce')
    df['SIPRI TIV for total order'] = pd.to_numeric(df['SIPRI TIV for total order'], errors='coerce')
    df['SIPRI TIV of delivered weapons'] = pd.to_numeric(df['SIPRI TIV of delivered weapons'], errors='coerce')

    # Add ISO codes for countries
    df['Recipient ISO'] = df['Recipient'].apply(get_iso_code)
    df['Supplier ISO'] = df['Supplier'].apply(get_iso_code)

    # Process delivery years
    def extract_years(year_string):
        if pd.isna(year_string):
            return []
        # Split on semicolon and remove any spaces
        years = [y.strip() for y in str(year_string).split(';')]
        # Convert to integers, handling empty strings
        return [int(y) for y in years if y.strip()]

    df['Delivery Years'] = df['Year(s) of delivery'].apply(extract_years)
    df['First Delivery Year'] = df['Delivery Years'].apply(lambda x: min(x) if x else None)
    df['Last Delivery Year'] = df['Delivery Years'].apply(lambda x: max(x) if x else None)

    # Calculate additional metrics
    df['Delivery Duration'] = df['Last Delivery Year'] - df['First Delivery Year']
    df['Delivery Complete'] = df['Number delivered'] >= df['Number ordered']

    # Save the processed data
    df.to_csv('data/sipri_trade_data_processed.csv', index=False)
    print("Saved processed data to sipri_trade_data_processed.csv")

    return df

if __name__ == "__main__":
    process_trade_data()
