import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# G20 countries with ISO codes
G20_COUNTRIES = {
    "Argentina": "ARG", 
    "Australia": "AUS", 
    "Brazil": "BRA", 
    "Canada": "CAN", 
    "China": "CHN", 
    "France": "FRA", 
    "Germany": "DEU", 
    "India": "IND", 
    "Indonesia": "IDN", 
    "Italy": "ITA", 
    "Japan": "JPN", 
    "Mexico": "MEX", 
    "Russia": "RUS", 
    "Saudi Arabia": "SAU", 
    "South Africa": "ZAF", 
    "Korea, Rep.": "KOR", 
    "Turkey": "TUR", 
    "United Kingdom": "GBR", 
    "United States": "USA"
}

# Alternative names that might appear in the dataset
G20_ALTERNATIVE_NAMES = {
    "Korea, Rep.": ["South Korea", "Republic of Korea"],
    "Russia": ["Russian Federation"],
    "United Kingdom": ["UK", "Great Britain"],
    "United States": ["USA", "US"]
}

# Reverse mapping from ISO to standard name for visualization
ISO_TO_NAME = {iso: name for name, iso in G20_COUNTRIES.items()}

# Primary data loading function - reads the JSON file containing merged SIPRI and World Bank data
def load_data(filepath='data/all_data_merged.json'):
    """Load the merged dataset from JSON file"""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Extracts only G20 countries from the full dataset for focused analysis
def extract_g20_countries(data):
    """Extract G20 countries from the dataset using ISO codes"""
    g20_data = []
    
    # Create a lookup set of G20 ISO codes
    g20_iso_set = set(G20_COUNTRIES.values())
    
    # Create a lookup set with all possible G20 country names
    g20_names_set = set(G20_COUNTRIES.keys())
    for country, alternatives in G20_ALTERNATIVE_NAMES.items():
        g20_names_set.update(alternatives)
    
    for country in data.get('countries', []):
        # First try matching by ISO code (more reliable)
        if country['ISO'] in g20_iso_set:
            country_copy = country.copy()
            # Store the standard name based on ISO
            country_copy['std_name'] = ISO_TO_NAME.get(country['ISO'], country['name'])
            g20_data.append(country_copy)
        # If ISO doesn't match, try matching by name as fallback
        elif country['name'] in g20_names_set:
            # Map alternative names to standard G20 name
            std_name = country['name']
            for g20_name, alternatives in G20_ALTERNATIVE_NAMES.items():
                if country['name'] in alternatives:
                    std_name = g20_name
                    break
            
            country_copy = country.copy()
            country_copy['std_name'] = std_name
            # If we can determine the ISO from the name, use that
            country_copy['std_ISO'] = G20_COUNTRIES.get(std_name, country['ISO'])
            g20_data.append(country_copy)
    
    return g20_data

# Transforms nested country time series data into a flat pandas DataFrame for analysis
def convert_to_dataframe(countries_data):
    """Convert the time series data for countries into a pandas DataFrame"""
    rows = []
    
    for country in countries_data:
        country_name = country.get('std_name', country['name'])
        country_iso = country.get('std_ISO', country['ISO'])
        
        for year_data in country['time_series']:
            row = {'country': country_name, 'ISO': country_iso, 'year': year_data['year']}
            row.update(year_data)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

# Creates a visual comparison between two indicators with country labels
def create_scatter_plot(df, x_indicator, y_indicator, year=None, countries=None, title=None, use_iso=True):
    """
    Create a scatter plot of two indicators for countries
    
    Parameters:
    df (DataFrame): DataFrame with country data
    x_indicator (str): Name of the indicator for x-axis
    y_indicator (str): Name of the indicator for y-axis
    year (int): Optional specific year to visualize
    countries (list): Optional list of specific country ISO codes to include
    title (str): Optional title for the plot
    use_iso (bool): Whether to use ISO codes or country names for identification
    """
    if year is not None:
        df = df[df['year'] == year]
    
    if countries is not None:
        id_column = 'ISO' if use_iso else 'country'
        df = df[df[id_column].isin(countries)]
    
    if x_indicator not in df.columns or y_indicator not in df.columns:
        print(f"Error: Indicators {x_indicator} or {y_indicator} not found in data")
        return
    
    # Drop rows with missing values for the selected indicators
    plot_df = df.dropna(subset=[x_indicator, y_indicator])
    
    if plot_df.empty:
        print("No data available for the selected indicators and filters")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot using ISO for the hue
    scatter_plot = sns.scatterplot(data=plot_df, x=x_indicator, y=y_indicator, 
                                  hue='ISO' if use_iso else 'country', s=100)
    
    # Add labels to points
    for _, row in plot_df.iterrows():
        label = row['ISO'] if use_iso else row['country']
        plt.annotate(label, 
                    (row[x_indicator], row[y_indicator]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center')
    
    plt.xlabel(x_indicator)
    plt.ylabel(y_indicator)
    
    if year:
        title_suffix = f" ({year})"
    else:
        title_suffix = ""
        
    plt.title(title or f"Relationship between {x_indicator} and {y_indicator}{title_suffix}")
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Update legend with country names if needed
    if not use_iso:
        handles, labels = scatter_plot.get_legend_handles_labels()
        plt.legend(handles=handles, labels=labels)
    else:
        handles, labels = scatter_plot.get_legend_handles_labels()
        plt.legend(handles=handles, labels=[ISO_TO_NAME.get(iso, iso) for iso in labels])
    
    plt.show()

# Calculates Pearson correlation coefficients between indicators for a specific country
def calculate_pearson_correlation(df, country_iso, indicators=None, use_iso=True):
    """
    Calculate Pearson correlation coefficients between indicators for a specific country
    
    Parameters:
    df (DataFrame): DataFrame with country data
    country_iso (str): Country ISO code
    indicators (list): Optional list of specific indicators to include
    use_iso (bool): Whether to use ISO codes or country names for identification
    
    Returns:
    DataFrame: Correlation matrix
    """
    id_column = 'ISO' if use_iso else 'country'
    country_df = df[df[id_column] == country_iso]
    
    if indicators:
        indicators = [ind for ind in indicators if ind in country_df.columns]
    else:
        # Exclude non-numeric columns
        indicators = [col for col in country_df.columns 
                     if col not in ['country', 'ISO', 'year'] 
                     and pd.api.types.is_numeric_dtype(country_df[col])]
    
    # Extract only the indicators we're interested in
    correlation_df = country_df[indicators].copy()
    
    # Drop rows with all NaN values
    correlation_df = correlation_df.dropna(how='all')
    
    if correlation_df.empty:
        country_name = ISO_TO_NAME.get(country_iso, country_iso) if use_iso else country_iso
        print(f"No valid data found for {country_name}")
        return None
    
    # Calculate correlation matrix
    corr_matrix = correlation_df.corr(method='pearson')
    
    return corr_matrix

# Creates a visual heatmap of correlation coefficients for a country
def visualize_correlation_matrix(corr_matrix, country_iso, title=None, use_iso=True):
    """
    Visualize a correlation matrix as a heatmap
    
    Parameters:
    corr_matrix (DataFrame): Correlation matrix
    country_iso (str): Country ISO code
    title (str): Optional custom title
    use_iso (bool): Whether to use ISO codes or country names for identification
    """
    if corr_matrix is None:
        return
    
    plt.figure(figsize=(14, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5)
    
    country_name = ISO_TO_NAME.get(country_iso, country_iso) if use_iso else country_iso
    plt.title(title or f"Correlation Matrix for {country_name}")
    plt.tight_layout()
    plt.show()

# Helper function to identify top-performing countries for a specific metric
def get_top_countries_by_metric(df, metric, year=None, top_n=4, ascending=False, use_iso=True):
    """
    Get the top N countries based on a specific metric
    
    Parameters:
    df (DataFrame): DataFrame with country data
    metric (str): Metric to sort by
    year (int): Optional specific year to consider
    top_n (int): Number of top countries to return
    ascending (bool): Sort order
    use_iso (bool): Whether to return ISO codes or country names
    
    Returns:
    list: List of top country identifiers (ISO or name)
    """
    if year:
        filtered_df = df[df['year'] == year]
    else:
        # Get the most recent year for each country
        filtered_df = df.sort_values('year', ascending=False).drop_duplicates('ISO')
    
    if metric not in filtered_df.columns:
        print(f"Error: Metric {metric} not found in data")
        return []
    
    # Drop countries with missing values for the metric
    filtered_df = filtered_df.dropna(subset=[metric])
    
    # Sort and get top N
    id_column = 'ISO' if use_iso else 'country'
    top_countries = filtered_df.sort_values(by=metric, ascending=ascending)[id_column].head(top_n).tolist()
    
    return top_countries

# Prepares country data for dimensionality reduction by filtering and cleaning the dataset
def prepare_for_dimensionality_reduction(df, indicators=None, countries=None, min_year=None, max_year=None, use_iso=True):
    """
    Prepare data for dimensionality reduction by extracting a clean matrix
    
    Parameters:
    df (DataFrame): DataFrame with country data
    indicators (list): List of indicators to include
    countries (list): Optional list of country ISO codes to include
    min_year (int): Optional minimum year to include
    max_year (int): Optional maximum year to include
    use_iso (bool): Whether to use ISO codes or country names for identification
    
    Returns:
    tuple: (data_matrix, feature_names, row_labels)
    """
    # Filter by countries if specified
    if countries:
        id_column = 'ISO' if use_iso else 'country'
        df = df[df[id_column].isin(countries)]
    
    # Filter by year range if specified
    if min_year:
        df = df[df['year'] >= min_year]
    if max_year:
        df = df[df['year'] <= max_year]
    
    # If no indicators specified, use all numeric columns except metadata
    if not indicators:
        indicators = [col for col in df.columns 
                    if col not in ['country', 'ISO', 'year'] 
                    and pd.api.types.is_numeric_dtype(df[col])]
    else:
        # Ensure all specified indicators exist in the dataframe
        indicators = [ind for ind in indicators if ind in df.columns]
    
    # Create row labels with country identifier and year
    id_column = 'ISO' if use_iso else 'country'
    df['row_label'] = df[id_column] + " (" + df['year'].astype(str) + ")"
    
    # Extract only the indicators we're interested in
    data_df = df[indicators].copy()
    
    # Drop rows with any missing values
    complete_indices = data_df.dropna().index
    data_df = data_df.loc[complete_indices]
    row_labels = df.loc[complete_indices, 'row_label'].tolist()
    
    # Convert to numpy array for dimensionality reduction
    data_matrix = data_df.to_numpy()
    
    return data_matrix, indicators, row_labels

# Performs Principal Component Analysis to reduce data dimensions while preserving variance
def perform_pca(data_matrix, n_components=2):
    """
    Perform PCA dimensionality reduction
    
    Parameters:
    data_matrix (numpy.ndarray): Matrix of data for dimensionality reduction
    n_components (int): Number of components to reduce to
    
    Returns:
    tuple: (reduced_data, explained_variance_ratio)
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)
    
    # Initialize PCA with the specified number of components
    pca = PCA(n_components=n_components)
    
    # Fit PCA to the data and transform
    reduced_data = pca.fit_transform(scaled_data)
    
    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    
    return reduced_data, explained_variance_ratio

# Performs t-SNE dimensionality reduction for non-linear data visualization
def perform_tsne(data_matrix, n_components=2, perplexity=30, n_iter=1000, random_state=42):
    """
    Perform t-SNE dimensionality reduction
    
    Parameters:
    data_matrix (numpy.ndarray): Matrix of data for dimensionality reduction
    n_components (int): Number of components to reduce to
    perplexity (float): The perplexity parameter for t-SNE
    n_iter (int): Number of iterations for optimization
    random_state (int): Random seed for reproducibility
    
    Returns:
    numpy.ndarray: Reduced data
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)
    
    # Initialize t-SNE with the specified parameters
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state
    )
    
    # Fit t-SNE to the data and transform
    reduced_data = tsne.fit_transform(scaled_data)
    
    return reduced_data

# Performs UMAP dimensionality reduction, which often preserves global structure better than t-SNE
def perform_umap_reduction(data_matrix, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Perform UMAP dimensionality reduction
    
    Parameters:
    data_matrix (numpy.ndarray): Matrix of data for dimensionality reduction
    n_components (int): Number of components to reduce to
    n_neighbors (int): Number of neighbors to consider
    min_dist (float): Minimum distance parameter for UMAP
    random_state (int): Random seed for reproducibility
    
    Returns:
    numpy.ndarray: Reduced data
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)
    
    # Initialize UMAP with the specified parameters
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    # Fit UMAP to the data and transform
    reduced_data = reducer.fit_transform(scaled_data)
    
    return reduced_data

# Creates visualizations of the dimensionally reduced data with labeled points
def visualize_reduced_data(reduced_data, row_labels, title=None, technique='PCA', show_country_names=False):
    """
    Visualize the reduced data in a scatter plot
    
    Parameters:
    reduced_data (numpy.ndarray): The reduced data from a dimensionality reduction technique
    row_labels (list): Labels for each data point (ISO codes with year)
    title (str): Optional title for the plot
    technique (str): The technique used for dimensionality reduction (for the title)
    show_country_names (bool): Whether to show full country names instead of ISO codes
    """
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=50, alpha=0.7)
    
    # Add labels for each point
    for i, label in enumerate(row_labels):
        # If needed, convert ISO codes to country names
        if show_country_names and "(" in label:
            iso_code = label.split(" (")[0]
            year = label.split("(")[1].rstrip(")")
            if iso_code in ISO_TO_NAME:
                label = f"{ISO_TO_NAME[iso_code]} ({year})"
        
        plt.annotate(label, 
                    (reduced_data[i, 0], reduced_data[i, 1]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center',
                    fontsize=8)
    
    plt.title(title or f"{technique} Visualization")
    plt.xlabel(f"{technique} Component 1")
    plt.ylabel(f"{technique} Component 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# Example execution section to demonstrate usage
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # Load the merged dataset
    print("Loading data...")
    data = load_data()
    
    # Extract G20 countries for analysis
    print("Extracting G20 countries...")
    g20_data = extract_g20_countries(data)
    print(f"Found {len(g20_data)} G20 countries in the dataset")
    
    # Convert to DataFrame for easier analysis
    print("Converting to DataFrame...")
    df = convert_to_dataframe(g20_data)
    print(f"DataFrame shape: {df.shape}")
    
    # Example 1: Create a scatter plot of military expenditure vs GDP
    print("\nExample 1: Creating scatter plot")
    mil_exp_indicator = "Military expenditure (% of GDP)"
    gdp_indicator = "GDP (current US$)"
    analysis_year = 2019  # Choose a recent year with good data coverage
    create_scatter_plot(df, mil_exp_indicator, gdp_indicator, year=analysis_year, 
                         title="Military Expenditure vs GDP for G20 Countries")
    
    # Example 2: Get top military spenders
    print("\nExample 2: Finding top military spenders")
    top_spenders = get_top_countries_by_metric(df, mil_exp_indicator, year=analysis_year, top_n=5)
    print(f"Top 5 G20 military spenders (% of GDP): {[ISO_TO_NAME.get(iso, iso) for iso in top_spenders]}")
    
    # Example 3: Calculate correlations for a specific country
    print("\nExample 3: Calculating correlations for USA")
    usa_corr = calculate_pearson_correlation(df, "USA", 
                                            indicators=[mil_exp_indicator, gdp_indicator,  
                                                       "Population, total", "Arms exports (SIPRI trend indicator values)"])
    print("Correlation matrix for USA:")
    print(usa_corr)
    
    # Example 4: Dimensionality reduction with PCA
    print("\nExample 4: Running PCA dimensionality reduction")
    # Select key economic and military indicators
    key_indicators = [
        mil_exp_indicator,
        gdp_indicator,
        "Arms exports (SIPRI trend indicator values)",
        "Arms imports (SIPRI trend indicator values)",
        "GDP per capita (current US$)",
        "Population, total"
    ]
    
    # Prepare data for top military spenders over recent years
    matrix, features, labels = prepare_for_dimensionality_reduction(
        df, indicators=key_indicators, countries=top_spenders, min_year=2010
    )
    
    if matrix.shape[0] > 0:
        # Perform PCA
        reduced_data, explained_variance = perform_pca(matrix)
        print(f"PCA explained variance: {explained_variance}")
        
        # Visualize PCA results
        visualize_reduced_data(reduced_data, labels, 
                              title="PCA Analysis of Top Military Spenders (2010-present)", 
                              show_country_names=True)
    else:
        print("Not enough data available for PCA analysis with selected parameters")
    
    print("\nAnalysis complete!")
