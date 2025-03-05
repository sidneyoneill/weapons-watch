import pandas as pd
import numpy as np
import pycountry
import geopandas as gpd


# Load the cleaned wide-format dataset
df = pd.read_csv('data/sipri_milex_data.csv')

# Melt the DataFrame so that each row is a (Country, Year, Expenditure) observation.
# 'Country' is kept as the identifier, and all other columns become part of the 'Year' variable.
tidy_df = pd.melt(df, id_vars=['Country'], var_name='Year', value_name='Expenditure')

# Convert the 'Year' column to numeric (it was originally the column names)
tidy_df['Year'] = pd.to_numeric(tidy_df['Year'], errors='coerce')

# Replace '...' and 'XXX' with NaN, since they indicate missing data or non-existent entries.
tidy_df['Expenditure'] = tidy_df['Expenditure'].replace({'...': np.nan, 'XXX': np.nan})

# Convert the 'Expenditure' column to a numeric type
tidy_df['Expenditure'] = pd.to_numeric(tidy_df['Expenditure'], errors='coerce')

# Optional: Drop rows where there is no expenditure data, if you don't need those entries.
tidy_df = tidy_df.dropna(subset=['Expenditure'])

# Reset index for a clean DataFrame
tidy_df = tidy_df.reset_index(drop=True)

# 1. Add ISO Codes to Your DataFrame

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

# Apply the function to create a new column 'ISO_Code'
tidy_df['ISO_Code'] = tidy_df['Country'].apply(get_iso_code)

# Check whether the function was successful, print unique countries
print(tidy_df[tidy_df['ISO_Code'].isna()]['Country'].unique())

# Optionally, drop rows where ISO code lookup failed
tidy_df = tidy_df.dropna(subset=['ISO_Code'])

print(tidy_df[tidy_df['ISO_Code'].isna()]['Country'].unique())

# Save the tidied DataFrame to a new CSV file
tidy_df.to_csv('data/sipri_milex_data_tidy.csv', index=False)

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
tidy_df_year = tidy_df[tidy_df['Year'] == year_to_map]

# Merge the GeoDataFrame with your expenditure data.
# Here we assume the GeoJSON has an ISO code column named 'ISO_A3'
merged_data = world.merge(tidy_df_year, left_on='ISO_A3', right_on='ISO_Code', how='left')

# Inspect the merged data to verify the join
print(merged_data.head())

# Inspect the columns of the merged data
print(merged_data.columns)

# Save the merged data to a new GeoJSON file
merged_data.to_file('data/sipri_milex_data_merged.geojson', driver='GeoJSON')
