# backend/prepare_arms_trade_matrix.py

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional

# Define the target countries for our chord diagram
TARGET_COUNTRIES = ['USA', 'RUSSIA', 'UK', 'GERMANY', 'CHINA', 'ISRAEL']

def standardize_country_name(country_name: str) -> str:
    """
    Standardize country names to match our target countries list.
    """
    country_mapping = {
        'UNITED STATES': 'USA',
        'UNITED STATES OF AMERICA': 'USA',
        'UNITED KINGDOM': 'UK',
        'GREAT BRITAIN': 'UK',
        'RUSSIAN FEDERATION': 'RUSSIA',
        "PEOPLE'S REPUBLIC OF CHINA": 'CHINA',
        'PRC': 'CHINA',
        'FEDERAL REPUBLIC OF GERMANY': 'GERMANY',
    }
    
    # Convert to uppercase for consistent matching
    country_upper = country_name.upper()
    
    # Return the mapped name if it exists, otherwise return the original name
    return country_mapping.get(country_upper, country_upper)

def prepare_arms_trade_matrix(year: int) -> Dict[str, Any]:
    """
    Prepare arms trade data for the chord diagram for a specific year.
    
    Args:
        year: The year to filter the data for
        
    Returns:
        A dictionary containing the countries list and the trade matrix
    """
    # Load the arms trade data
    try:
        df = pd.read_csv('../data/sipri_trade_data_tidy.csv')
    except FileNotFoundError:
        # Try the original file if the tidy version doesn't exist
        df = pd.read_csv('../data/sipri_trade_data.csv', encoding='latin-1')
    
    # Filter by the selected year
    # First try to use the Delivery Year Numeric column if it exists
    if 'Delivery Year Numeric' in df.columns:
        year_filtered_df = df[df['Delivery Year Numeric'] == year]
    # Otherwise, try to extract the year from the Year(s) of delivery column
    elif 'Year(s) of delivery' in df.columns:
        year_filtered_df = df[df['Year(s) of delivery'].str.contains(str(year), na=False)]
    else:
        raise ValueError("Could not find year column in the dataset")
    
    # If no data for the selected year, return empty matrix
    if year_filtered_df.empty:
        countries = TARGET_COUNTRIES + ['OTHER']
        empty_matrix = np.zeros((len(countries), len(countries)))
        return {
            'countries': countries,
            'matrix': empty_matrix.tolist()
        }
    
    # Standardize country names and categorize non-target countries as 'OTHER'
    year_filtered_df['Supplier_Std'] = year_filtered_df['Supplier'].apply(standardize_country_name)
    year_filtered_df['Recipient_Std'] = year_filtered_df['Recipient'].apply(standardize_country_name)
    
    # Categorize non-target countries as 'OTHER'
    year_filtered_df['Supplier_Cat'] = year_filtered_df['Supplier_Std'].apply(
        lambda x: x if x in TARGET_COUNTRIES else 'OTHER'
    )
    year_filtered_df['Recipient_Cat'] = year_filtered_df['Recipient_Std'].apply(
        lambda x: x if x in TARGET_COUNTRIES else 'OTHER'
    )
    
    # Use SIPRI TIV as the value metric
    # First check which column is available
    if 'SIPRI TIV for total order' in df.columns:
        value_col = 'SIPRI TIV for total order'
    elif 'SIPRI TIV of delivered weapons' in df.columns:
        value_col = 'SIPRI TIV of delivered weapons'
    else:
        raise ValueError("Could not find SIPRI TIV column in the dataset")
    
    # Aggregate flows by supplier-recipient
    aggregated_df = year_filtered_df.groupby(['Supplier_Cat', 'Recipient_Cat'])[value_col].sum().reset_index()
    
    # Create the matrix
    countries = TARGET_COUNTRIES + ['OTHER']
    matrix = np.zeros((len(countries), len(countries)))
    
    for _, row in aggregated_df.iterrows():
        supplier = row['Supplier_Cat']
        recipient = row['Recipient_Cat']
        value = row[value_col]
        
        supplier_idx = countries.index(supplier)
        recipient_idx = countries.index(recipient)
        
        matrix[supplier_idx][recipient_idx] = value
    
    return {
        'countries': countries,
        'matrix': matrix.tolist()
    }

def generate_arms_trade_matrices(start_year: int = 2015, end_year: int = 2021) -> None:
    """
    Generate arms trade matrices for a range of years and save them to JSON files.
    
    Args:
        start_year: The first year to generate a matrix for
        end_year: The last year to generate a matrix for
    """
    # Create output directory if it doesn't exist
    output_dir = '../data/arms_trade_matrices'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a matrix for each year in the range
    for year in range(start_year, end_year + 1):
        matrix_data = prepare_arms_trade_matrix(year)
        
        # Save to a JSON file
        output_file = f'{output_dir}/arms_trade_matrix_{year}.json'
        with open(output_file, 'w') as f:
            json.dump(matrix_data, f, indent=2)
        
        print(f"Generated arms trade matrix for {year} and saved to {output_file}")
    
    # Generate an all-years file with a dictionary of matrices by year
    all_matrices = {}
    for year in range(start_year, end_year + 1):
        all_matrices[str(year)] = prepare_arms_trade_matrix(year)
    
    # Save all matrices to a single JSON file
    all_years_file = f'{output_dir}/arms_trade_matrices_all_years.json'
    with open(all_years_file, 'w') as f:
        json.dump(all_matrices, f, indent=2)
    
    print(f"Generated combined arms trade matrices for all years and saved to {all_years_file}")

def main():
    """
    Main function to generate arms trade matrices.
    """
    print("Generating arms trade matrices for chord diagram...")
    generate_arms_trade_matrices()
    print("Done!")

if __name__ == '__main__':
    main()