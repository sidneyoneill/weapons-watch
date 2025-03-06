import pandas as pd
import numpy as np
import pycountry
import geopandas as gpd


# Load the cleaned wide-format dataset
df_absolute = pd.read_csv('data/sipri_milex_data.csv')  # First csv: absolute values
df_percentage = pd.read_csv('data/sipri_milex_gdp_data.csv')  # Second csv: percentage of GDP

# Custom mapping for country names that don't match pycountry's expected names
custom_mappings = {
    "Korea, South": "KOR",                  # South Korea
    "Cote d'Ivoire": "CIV",                 # Ivory Coast (Côte d'Ivoire)
    "Congo, DR": "COD",                     # Democratic Republic of the Congo
    "Congo, Republic": "COG",               # Republic of the Congo
    "Brunei": "BRN",                        # Brunei Darussalam
    "Gambia, The": "GMB",                   # The Gambia
    "Cape Verde": "CPV",                    # Cabo Verde
    "Russia": "RUS",                        # Russian Federation
    "Timor Leste": "TLS",                   # Timor-Leste
    "Kosovo": "XKX",                        # Custom code for Kosovo
    # Historical or non-standard entities
    "Czechoslovakia": None,
    "German Democratic Republic": None,
    "USSR": None,
    "Yemen, North": None
}

def get_iso_code(country_name):
    """
    Lookup ISO Alpha-3 country code using a custom mapping first, then pycountry.
    Returns None if no match is found.
    """
    # Check custom mapping first
    if country_name in custom_mappings:
        return custom_mappings[country_name]
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except LookupError:
        return None

def create_tidy_df(df, value_name):
    """
    Process a wide-format dataframe into a tidy format with ISO codes
    
    Args:
        df: Input dataframe in wide format
        value_name: Name for the value column (e.g., 'Expenditure', 'ExpenditureGDP')
    
    Returns:
        A tidy dataframe with ISO codes
    """
    # Melt the DataFrame
    tidy_df = pd.melt(df, id_vars=['Country'], var_name='Year', value_name=value_name)
    
    # Convert the 'Year' column to numeric
    tidy_df['Year'] = pd.to_numeric(tidy_df['Year'], errors='coerce')
    
    # Replace '...' and 'XXX' with NaN
    tidy_df[value_name] = tidy_df[value_name].replace({'...': np.nan, 'XXX': np.nan})
    
    # Convert the value column to numeric
    tidy_df[value_name] = pd.to_numeric(tidy_df[value_name], errors='coerce')
    
    # Drop rows with no expenditure data
    tidy_df = tidy_df.dropna(subset=[value_name])
    
    # Reset index
    tidy_df = tidy_df.reset_index(drop=True)
    
    # Apply the function to create a new column 'ISO_Code'
    tidy_df['ISO_Code'] = tidy_df['Country'].apply(get_iso_code)
    
    # Drop rows where ISO code lookup failed
    tidy_df = tidy_df.dropna(subset=['ISO_Code'])
    
    return tidy_df

# Create tidy dataframes for both absolute and percentage data
tidy_df_absolute = create_tidy_df(df_absolute, 'Expenditure')
tidy_df_percentage = create_tidy_df(df_percentage, 'Expenditure')

# Check which countries failed ISO code lookup
print("Countries without ISO codes in absolute data:")
print(df_absolute[~df_absolute['Country'].isin(tidy_df_absolute['Country'])]['Country'].unique())
print("\nCountries without ISO codes in percentage data:")
print(df_percentage[~df_percentage['Country'].isin(tidy_df_percentage['Country'])]['Country'].unique())

# Save the tidied DataFrames to new CSV files
tidy_df_absolute.to_csv('data/sipri_milex_data_tidy.csv', index=False)
tidy_df_percentage.to_csv('data/sipri_milex_gdp_data_tidy.csv', index=False)

# 2. Obtain and Load GeoJSON Data for Country Boundaries

# Load the GeoJSON file containing country boundaries.
# Ensure you have downloaded a GeoJSON file (e.g., 'world_countries.geojson')
world = gpd.read_file('data/world_countries.geojson')

# Inspect the GeoDataFrame to see which column contains ISO codes.
# Commonly, this column is named 'ISO_A3' or 'iso_a3'.
print(world.columns)

# 3. Merge the Expenditure Data with the Geo Data

# If you want to visualize the data for a specific year (for example, 2020):
year_to_map = 2020

# For absolute expenditure
tidy_df_year_abs = tidy_df_absolute[tidy_df_absolute['Year'] == year_to_map]
merged_data_abs = world.merge(tidy_df_year_abs, left_on='ISO_A3', right_on='ISO_Code', how='left')
print("\nMerged absolute data head:")
print(merged_data_abs.head())
merged_data_abs.to_file('data/sipri_milex_data_merged.geojson', driver='GeoJSON')

# For percentage of GDP
tidy_df_year_pct = tidy_df_percentage[tidy_df_percentage['Year'] == year_to_map]
merged_data_pct = world.merge(tidy_df_year_pct, left_on='ISO_A3', right_on='ISO_Code', how='left')
print("\nMerged percentage data head:")
print(merged_data_pct.head())
merged_data_pct.to_file('data/sipri_milex_gdp_data_merged.geojson', driver='GeoJSON')
