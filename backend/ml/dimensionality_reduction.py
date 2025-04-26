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
import os
import itertools  # Add this import

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

# Theme mapping for indicators
INDICATOR_THEMES = {
    # Military & Security indicators
    'military_expenditure': 'Military & Security',
    'military_expenditure_gdp': 'Military & Security',
    'arms_exports_sipri_trend_indicator_values': 'Military & Security',
    'arms_imports_sipri_trend_indicator_values': 'Military & Security',
    'armed_forces_personnel_total': 'Military & Security',
    
    # Economic indicators
    'gdp_growth_annual_percent': 'Economic',
    'foreign_direct_investment_net_outflows_percent_of_gdp': 'Economic',
    'tax_revenue_percent_of_gdp': 'Economic',
    'tariff_rate_applied_simple_mean_all_products_percent': 'Economic',
    'ores_and_metals_exports_percent_of_merchandise_exports': 'Economic',
    
    # Social Development indicators
    'income_share_held_by_highest_10percent': 'Inequality & Development',
    'income_share_held_by_lowest_10percent': 'Inequality & Development',
    'individuals_using_the_internet_percent_of_population': 'Inequality & Development',
    'rural_population_percent_of_total_population': 'Inequality & Development',
    
    # Political & Migration indicators
    'political_stability_and_absence_of_violenceterrorism_percentile_rank': 'Political & Migration',
    'refugee_population_by_country_or_territory_of_origin': 'Political & Migration',
    'internally_displaced_persons_total_displaced_by_conflict_and_violence_number_of_people': 'Political & Migration',
    'international_migrant_stock_percent_of_population': 'Political & Migration',
    
    # Energy & Environment indicators
    'fossil_fuel_energy_consumption_percent_of_total': 'Energy & Environment'
}

# Default theme for unmapped indicators
DEFAULT_THEME = 'Other'

# Theme colors for consistent visualization
THEME_COLORS = {
    'Military & Security': '#ff7f0e',       # Orange
    'Economic': '#1f77b4',                  # Blue
    'Inequality & Development': '#2ca02c',  # Green
    'Political & Migration': '#d62728',     # Red
    'Energy & Environment': '#9467bd',      # Purple
    'Other': '#8c564b'                      # Brown
}

class DimensionalityReducer:
    def __init__(self, data_path=None, data=None, g20_only=True):
        """
        Initialize the DimensionalityReducer class
        
        Parameters:
        data_path (str): Path to the JSON data file
        data (DataFrame): Optional pre-loaded pandas DataFrame
        g20_only (bool): Whether to filter for only G20 countries or use all countries
        """
        if data is not None:
            self.df = data
        elif data_path:
            self.data = self.load_data(data_path)
            
            if g20_only:
                # Filter for G20 countries only
                self.df = self.convert_to_dataframe(self.extract_g20_countries(self.data))
            else:
                # Use all countries
                self.df = self.convert_to_dataframe(self.extract_all_countries(self.data))
        else:
            raise ValueError("Either data_path or data must be provided")
    
    def load_data(self, filepath):
        """Load the merged dataset from JSON file"""
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    
    def extract_g20_countries(self, data):
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

    def extract_all_countries(self, data):
        """Extract all countries from the dataset"""
        # Simply return all countries without filtering
        return data.get('countries', [])
    
    def convert_to_dataframe(self, countries_data):
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
    
    def filter_years(self, start_year, end_year):
        """Filter the DataFrame to include only the specified range of years"""
        self.df = self.df[(self.df['year'] >= start_year) & (self.df['year'] <= end_year)]
        return self
    
    def check_missing_data(self):
        """Check for missing data in the DataFrame and highlight potential issues"""
        missing_values = self.df.isnull().sum()
        
        if missing_values.sum() == 0:
            print("No missing values found in the dataset")
        else:
            print("Missing values found in the dataset:")
            print(missing_values[missing_values > 0])
        
        return missing_values[missing_values > 0]
    
    def prepare_for_dimensionality_reduction(self, indicators=None, countries=None, 
                                           use_iso=True, impute_missing=True, 
                                           min_non_null_ratio=0.7):
        """
        Prepare data for dimensionality reduction by extracting a clean matrix
        
        Parameters:
        indicators (list): List of indicators to include
        countries (list): Optional list of country ISO codes to include
        use_iso (bool): Whether to use ISO codes or country names for identification
        impute_missing (bool): Whether to impute missing values instead of dropping rows
        min_non_null_ratio (float): Minimum ratio of non-null values needed for an indicator to be kept
        
        Returns:
        tuple: (data_matrix, feature_names, row_labels)
        """
        # Make a copy to avoid modifying the original dataframe
        work_df = self.df.copy()
        
        # Filter by countries if specified
        if countries:
            id_column = 'ISO' if use_iso else 'country'
            work_df = work_df[work_df[id_column].isin(countries)]
        
        # If no indicators specified, use all numeric columns except metadata
        if not indicators:
            all_possible_indicators = [col for col in work_df.columns 
                        if col not in ['country', 'ISO', 'year', 'row_label'] 
                        and pd.api.types.is_numeric_dtype(work_df[col])]
                        
            # Filter out indicators with too many missing values
            indicators = []
            for col in all_possible_indicators:
                non_null_ratio = work_df[col].notna().mean()
                if non_null_ratio >= min_non_null_ratio:
                    indicators.append(col)
                    
            if not indicators:
                print(f"Warning: No indicators meet the minimum non-null ratio of {min_non_null_ratio}. Try lowering this threshold.")
                # Fall back to using the least-sparse indicators
                if all_possible_indicators:
                    non_null_counts = {col: work_df[col].notna().mean() for col in all_possible_indicators}
                    sorted_indicators = sorted(non_null_counts.items(), key=lambda x: x[1], reverse=True)
                    indicators = [ind for ind, _ in sorted_indicators[:10]]  # Take top 10
                    print(f"Using {len(indicators)} indicators with highest coverage.")
        else:
            # Ensure all specified indicators exist in the dataframe
            indicators = [ind for ind in indicators if ind in work_df.columns]
        
        print(f"Number of indicators selected: {len(indicators)}")
        
        # Create row labels with country identifier and year
        id_column = 'ISO' if use_iso else 'country'
        work_df['row_label'] = work_df[id_column] + " (" + work_df['year'].astype(str) + ")"
        
        # Extract only the indicators we're interested in
        data_df = work_df[indicators].copy()
        
        # Handle missing values
        if impute_missing:
            # Simple imputation (interpolation) with the midpoint between the previous and next valid values
            data_df = data_df.interpolate(limit_direction='both')
            # Fill any remaining missing values with the column mean
            data_df = data_df.fillna(data_df.mean())
            complete_indices = data_df.index
            print(f"Missing values imputed with interpolation and column means.")
        else:
            # Drop rows with any missing values
            complete_indices = data_df.dropna().index
            data_df = data_df.loc[complete_indices]
            
        # Get row labels for the final dataset
        row_labels = work_df.loc[complete_indices, 'row_label'].tolist()
        country_codes = work_df.loc[complete_indices, 'ISO'].tolist()
        years = work_df.loc[complete_indices, 'year'].tolist()
        
        print(f"Final dataset has {len(row_labels)} rows with complete data.")
        
        # Convert to numpy array for dimensionality reduction
        if len(data_df) > 0:
            data_matrix = data_df.to_numpy()
        else:
            print("Warning: No complete rows found. Returning empty matrix.")
            data_matrix = np.zeros((0, len(indicators)))
        
        self.data_matrix = data_matrix
        self.indicators = indicators
        self.row_labels = row_labels
        self.country_codes = country_codes
        self.years = years
        
        return data_matrix, indicators, row_labels, country_codes, years
    
    def perform_pca(self, n_components=2):
        """
        Perform PCA dimensionality reduction
        
        Parameters:
        n_components (int): Number of components to reduce to (2 or 3 for visualization)
        
        Returns:
        tuple: (reduced_data, explained_variance_ratio, loadings)
        """
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data_matrix)
        
        # Initialize PCA with the specified number of components
        pca = PCA(n_components=n_components)
        
        # Fit PCA to the data and transform
        reduced_data = pca.fit_transform(scaled_data)
        
        # collect the loadings
        loadings = pca.components_

        # Get the explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        
        self.pca_results = {
            'reduced_data': reduced_data,
            'explained_variance_ratio': explained_variance_ratio,
            'loadings': loadings
        }
        
        print(f"PCA explained variance: {explained_variance_ratio}")
        
        return reduced_data, explained_variance_ratio, loadings
    
    def perform_tsne(self, n_components=2, perplexity=30, n_iter=1000, random_state=42):
        """
        Perform t-SNE dimensionality reduction
        
        Parameters:
        n_components (int): Number of components to reduce to
        perplexity (float): The perplexity parameter for t-SNE
        n_iter (int): Number of iterations for optimization
        random_state (int): Random seed for reproducibility
        
        Returns:
        numpy.ndarray: Reduced data
        """
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data_matrix)
        
        # Initialize t-SNE with the specified parameters
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state
        )
        
        # Fit t-SNE to the data and transform
        reduced_data = tsne.fit_transform(scaled_data)
        
        self.tsne_results = {
            'reduced_data': reduced_data
        }
        
        print(f"t-SNE completed with {n_components} components")
        
        return reduced_data
    
    def visualize_reduced_data(self, reduced_data=None, technique='PCA', title=None, 
                             show_country_names=True, plot_3d=False, save_path=None,
                             display_plot=True):
        """
        Visualize the reduced data in a scatter plot with countries colored consistently
        
        Parameters:
        reduced_data (numpy.ndarray): The reduced data from a dimensionality reduction technique
        technique (str): The technique used for dimensionality reduction (for the title)
        title (str): Optional title for the plot
        show_country_names (bool): Whether to show full country names instead of ISO codes
        plot_3d (bool): Whether to create a 3D plot (requires reduced_data to have 3 columns)
        save_path (str): Optional path to save the visualization
        display_plot (bool): Whether to display the plot interactively
        """
        if reduced_data is None:
            # Try to get results from previous runs - normalize technique name for attribute lookups
            technique_attr = technique.lower().replace('-', '')  # Convert 't-SNE' to 'tsne'
            
            if technique_attr == 'pca' and hasattr(self, 'pca_results'):
                reduced_data = self.pca_results['reduced_data']
            elif technique_attr == 'tsne' and hasattr(self, 'tsne_results'):
                reduced_data = self.tsne_results['reduced_data']
            elif technique_attr == 'umap' and hasattr(self, 'umap_results'):
                reduced_data = self.umap_results['reduced_data']
            else:
                raise ValueError(f"No {technique} results found. Run perform_{technique_attr} first.")
        
        # Check if we have enough components for 3D plotting
        if plot_3d and reduced_data.shape[1] < 3:
            print("Warning: 3D plot requested but data has fewer than 3 components. Falling back to 2D.")
            plot_3d = False
        
        # Create a DataFrame for easier plotting
        plot_df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'country': self.country_codes,
            'year': self.years,
            'label': self.row_labels
        })
        
        if plot_3d:
            plot_df['z'] = reduced_data[:, 2]  # Add the third component
            
            # Create 3D plot
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get unique countries for consistent coloring
            unique_countries = list(set(self.country_codes))
            cmap = plt.cm.get_cmap('tab20' if len(unique_countries) <= 20 else 'rainbow')
            colors = {country: cmap(i/len(unique_countries)) for i, country in enumerate(unique_countries)}
            
            # Plot each country with its own color
            for country in unique_countries:
                country_data = plot_df[plot_df['country'] == country]
                ax.scatter(country_data['x'], country_data['y'], country_data['z'], 
                          label=country, 
                          alpha=0.7, 
                          s=80)
            
            # Add labels
            ax.set_xlabel(f"{technique} Component 1")
            ax.set_ylabel(f"{technique} Component 2")
            ax.set_zlabel(f"{technique} Component 3")
            
            # Use country names instead of ISO codes in the legend if requested
            if show_country_names:
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [ISO_TO_NAME.get(label, label) for label in labels]
                ax.legend(handles, new_labels)
            else:
                ax.legend()
            
            # Improve readability
            ax.grid(True)
            plt.title(title or f"{technique} 3D Visualization")
            
        else:
            # Create 2D plot (original behavior)
            plt.figure(figsize=(12, 10))
            scatter_plot = sns.scatterplot(
                data=plot_df,
                x='x', 
                y='y',
                hue='country',
                palette='tab20',
                s=100,
                alpha=0.7
            )
            
            # Use country names instead of ISO codes in the legend if requested
            if show_country_names:
                handles, labels = scatter_plot.get_legend_handles_labels()
                new_labels = [ISO_TO_NAME.get(label, label) for label in labels]
                plt.legend(handles, new_labels)
            
            plt.xlabel(f"{technique} Component 1")
            plt.ylabel(f"{technique} Component 2")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.title(title or f"{technique} 2D Visualization")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        if display_plot:
            plt.show()
        else:
            plt.close()
        
    def plot_pca_loadings(self, components=None, n_components=2, save_path=None, display_plot=True):
        """
        Visualize the PCA loadings to understand which features contribute to each component
        
        Parameters:
        components (numpy.ndarray): PCA components/loadings
        n_components (int): Number of components to visualize
        save_path (str): Optional path to save the visualization
        display_plot (bool): Whether to display the plot interactively
        """
        if components is None:
            if hasattr(self, 'pca_results'):
                components = self.pca_results['loadings']
            else:
                raise ValueError("No PCA results found. Run perform_pca first.")
        
        # Limit to requested number of components
        components = components[:n_components]
        
        plt.figure(figsize=(14, n_components * 4))
        
        for i, component in enumerate(components):
            plt.subplot(n_components, 1, i + 1)
            # Sort loadings by absolute value for better visualization
            indices = np.argsort(np.abs(component))[::-1]
            
            plt.barh(range(len(indices)), component[indices], color='skyblue')
            plt.yticks(range(len(indices)), [self.indicators[idx] for idx in indices])
            plt.title(f"Component {i+1} Loadings")
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA loadings visualization saved to {save_path}")
        
        if display_plot:
            plt.show()
        else:
            plt.close()

    def export_results(self, export_dir, prefix='dim_reduction'):
        """
        Export dimensionality reduction results to files
        
        Parameters:
        export_dir (str): Directory to save files
        prefix (str): Prefix for filenames
        """
        os.makedirs(export_dir, exist_ok=True)
        
        # Create metadata about the analysis
        metadata = {
            'indicators_used': self.indicators,
            'num_data_points': len(self.row_labels),
            'countries_included': list(set(self.country_codes))
        }
        
        # Export metadata
        with open(os.path.join(export_dir, f"{prefix}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Export PCA results if available
        if hasattr(self, 'pca_results'):
            pca_results = {
                'explained_variance_ratio': self.pca_results['explained_variance_ratio'].tolist(),
                'data': []
            }
            
            for i, label in enumerate(self.row_labels):
                country = self.country_codes[i]
                year = self.years[i]
                coords = self.pca_results['reduced_data'][i].tolist()
                
                pca_results['data'].append({
                    'country': country,
                    'country_name': ISO_TO_NAME.get(country, country),
                    'year': year,
                    'coordinates': coords
                })
            
            with open(os.path.join(export_dir, f"{prefix}_pca_results.json"), 'w') as f:
                json.dump(pca_results, f, indent=2)
        
        # Export t-SNE results if available
        if hasattr(self, 'tsne_results'):
            tsne_results = {'data': []}
            
            for i, label in enumerate(self.row_labels):
                country = self.country_codes[i]
                year = self.years[i]
                coords = self.tsne_results['reduced_data'][i].tolist()
                
                tsne_results['data'].append({
                    'country': country,
                    'country_name': ISO_TO_NAME.get(country, country),
                    'year': year,
                    'coordinates': coords
                })
            
            with open(os.path.join(export_dir, f"{prefix}_tsne_results.json"), 'w') as f:
                json.dump(tsne_results, f, indent=2)
        print(f"Results exported to {export_dir}")

    # save plots to export directory
    def save_plots(self, export_dir, prefix='dim_reduction'):
        """
        Save the PCA and t-SNE plots to the export directory without displaying them
        
        Parameters:
        export_dir (str): Directory to save files
        prefix (str): Prefix for filenames
        """
        os.makedirs(export_dir, exist_ok=True)
        
        # Save PCA plot if available
        if hasattr(self, 'pca_results'):
            pca_plot_path = os.path.join(export_dir, f"{prefix}_pca_plot.png")
            self.visualize_reduced_data(
                technique='PCA', 
                show_country_names=True, 
                save_path=pca_plot_path,
                display_plot=False
            )
            
            # Also save PCA loadings for 2 components
            pca_loadings_path = os.path.join(export_dir, f"{prefix}_pca_loadings_2d.png")
            self.plot_pca_loadings(
                save_path=pca_loadings_path,
                display_plot=False
            )

            # New themed visualizations
            try:
                # Themed loadings analysis
                themed_loadings_path = os.path.join(export_dir, f"{prefix}_pca_themed_loadings.png")
                self.analyze_themed_loadings(
                    n_components=min(3, len(self.pca_results['loadings'])), 
                    save_path=themed_loadings_path,
                    display_plot=False
                )
                
                # Themed biplot
                themed_biplot_path = os.path.join(export_dir, f"{prefix}_pca_themed_biplot.png")
                self.plot_themed_biplot(
                    save_path=themed_biplot_path,
                    display_plot=False
                )
            except Exception as e:
                print(f"Warning: Could not generate themed visualizations: {e}")


            # save pca loadings with 3 components
            pca_loadings_3d_path = os.path.join(export_dir, f"{prefix}_pca_loadings_3d.png")
            self.plot_pca_loadings(
                n_components=3,
                save_path=pca_loadings_3d_path,
                display_plot=False
            )


        # Save t-SNE plot if available
        if hasattr(self, 'tsne_results'):
            tsne_plot_path = os.path.join(export_dir, f"{prefix}_tsne_plot.png")
            self.visualize_reduced_data(
                technique='t-SNE', 
                show_country_names=True, 
                save_path=tsne_plot_path,
                display_plot=False
            )
        
        # Save 3D PCA plot if available and it has 3 components
        if hasattr(self, 'pca_results') and self.pca_results['reduced_data'].shape[1] >= 3:
            pca_3d_plot_path = os.path.join(export_dir, f"{prefix}_pca_3d_plot.png")
            self.visualize_reduced_data(
                technique='PCA',
                show_country_names=True,
                plot_3d=True,
                save_path=pca_3d_plot_path,
                display_plot=False
            )
        
        # Save 3D t-SNE plot if available and it has 3 components
        if hasattr(self, 'tsne_results') and self.tsne_results['reduced_data'].shape[1] >= 3:
            tsne_3d_plot_path = os.path.join(export_dir, f"{prefix}_tsne_3d_plot.png")
            self.visualize_reduced_data(
                technique='t-SNE',
                show_country_names=True,
                plot_3d=True,
                save_path=tsne_3d_plot_path,
                display_plot=False
            )

        print(f"Plots saved to {export_dir}")

    def export_thematic_insights(self, export_dir, prefix='dim_reduction'):
        """
        Export thematic insights from PCA analysis to a JSON file
        
        Parameters:
        export_dir (str): Directory to save files
        prefix (str): Prefix for filenames
        """
        os.makedirs(export_dir, exist_ok=True)
        
        if not hasattr(self, 'pca_results'):
            raise ValueError("No PCA results found. Run perform_pca first.")
        
        # Generate thematic analysis
        try:
            # Use display_plot=False to avoid showing plots during analysis
            component_themes = self.analyze_themed_loadings(
                n_components=min(3, len(self.pca_results['loadings'])),
                display_plot=False
            )
            
            # Create component descriptions using thematic information
            component_descriptions = {}
            for comp_name, comp_info in component_themes.items():
                dominant_theme = comp_info['dominant_theme']
                theme_percentages = comp_info['theme_percentages']
                
                # Format key indicators for this component
                key_indicators_text = []
                for theme, indicators in comp_info['key_indicators'].items():
                    indicator_texts = [f"{ind['Indicator']} ({ind['Loading']:.2f})" 
                                    for ind in indicators[:3]]  # Top 3 indicators
                    key_indicators_text.append(f"{theme}: {', '.join(indicator_texts)}")
                
                # Create a narrative description
                description = f"Component primarily represents {dominant_theme} ({theme_percentages[dominant_theme]:.1f}% of variance) "
                
                # Add secondary themes if they contribute significantly
                secondary_themes = [t for t, p in theme_percentages.items() 
                                if t != dominant_theme and p > 10.0]
                if secondary_themes:
                    description += f"with contributions from {', '.join(secondary_themes)}"
                
                # Add direction information
                direction = comp_info['direction'].get(dominant_theme, 'neutral')
                if direction == 'positive':
                    description += ". Higher values indicate stronger presence of these factors."
                elif direction == 'negative':
                    description += ". Lower values indicate stronger presence of these factors."
                
                component_descriptions[comp_name] = {
                    "description": description,
                    "key_indicators": key_indicators_text,
                    "thematic_composition": {k: f"{v:.1f}%" for k, v in theme_percentages.items()}
                }
            
            # Save thematic insights to JSON
            insights = {
                "component_themes": component_themes,
                "component_descriptions": component_descriptions,
                "themes_overview": {
                    theme: list(indicators) 
                    for theme, indicators in itertools.groupby(
                        sorted(INDICATOR_THEMES.items(), key=lambda x: x[1]), 
                        key=lambda x: x[1]
                    )
                }
            }
            
            insights_path = os.path.join(export_dir, f"{prefix}_thematic_insights.json")
            with open(insights_path, 'w') as f:
                json.dump(insights, f, indent=2)
            
            print(f"Thematic insights exported to {insights_path}")
            
        except Exception as e:
            print(f"Warning: Could not export thematic insights: {e}")
            
    def analyze_themed_loadings(self, n_components=2, display_plot=True, save_path=None):
        """
        Analyze and visualize PCA loadings grouped by themes
        
        Parameters:
        n_components (int): Number of components to analyze
        display_plot (bool): Whether to display the plot interactively
        save_path (str): Optional path to save the visualization
        
        Returns:
        dict: Information about thematic composition of each component
        """
        if not hasattr(self, 'pca_results'):
            raise ValueError("No PCA results found. Run perform_pca first.")
        
        # Get components and indicators
        components = self.pca_results['loadings'][:n_components]
        indicators = self.indicators
        
        # Create a figure with a subplot for each component
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 5 * n_components))
        if n_components == 1:
            axes = [axes]
        
        component_themes = {}
        
        for i, component in enumerate(components):
            # Create a DataFrame with indicators and their loadings for this component
            loadings_df = pd.DataFrame({
                'Indicator': indicators,
                'Loading': component,
                'Abs_Loading': np.abs(component)
            })
            
            # Sort by absolute loading value for importance
            loadings_df = loadings_df.sort_values('Abs_Loading', ascending=False)
            
            # Add theme information
            loadings_df['Theme'] = loadings_df['Indicator'].map(
                lambda x: INDICATOR_THEMES.get(x, DEFAULT_THEME)
            )
            
            # Get thematic contribution for this component
            # Sum the absolute loadings by theme to measure theme importance
            theme_importance = loadings_df.groupby('Theme')['Abs_Loading'].sum()
            total_importance = theme_importance.sum()
            theme_percentages = (theme_importance / total_importance * 100).round(1)
            
            # Sort themes by importance
            theme_percentages = theme_percentages.sort_values(ascending=False)
            
            # Store the dominant themes for this component
            component_themes[f"Component_{i+1}"] = {
                'dominant_theme': theme_percentages.index[0] if not theme_percentages.empty else 'Unknown',
                'theme_percentages': theme_percentages.to_dict(),
                'direction': {},
                'key_indicators': {}
            }
            
            # For each theme, determine if it contributes positively or negatively
            for theme in theme_percentages.index:
                theme_indicators = loadings_df[loadings_df['Theme'] == theme]
                
                # Get the sum of loadings (with signs) to determine direction
                direction_sum = theme_indicators['Loading'].sum()
                direction = 'positive' if direction_sum >= 0 else 'negative'
                
                # Store the direction and key indicators
                component_themes[f"Component_{i+1}"]['direction'][theme] = direction
                
                # Get top 3 indicators for this theme
                top_indicators = theme_indicators.head(3)[['Indicator', 'Loading']].to_dict('records')
                component_themes[f"Component_{i+1}"]['key_indicators'][theme] = top_indicators
            
            # Visualization for this component
            ax = axes[i]
            
            # Use consistent colors for themes
            theme_colors = {theme: THEME_COLORS.get(theme, '#333333') for theme in loadings_df['Theme'].unique()}
            
            # Plot the loadings, colored by theme
            bar_positions = np.arange(len(loadings_df))
            for theme in loadings_df['Theme'].unique():
                theme_mask = loadings_df['Theme'] == theme
                ax.bar(
                    bar_positions[theme_mask], 
                    loadings_df.loc[theme_mask, 'Loading'],
                    label=f"{theme} ({theme_percentages.get(theme, 0)}%)",
                    color=theme_colors[theme],
                    alpha=0.7
                )
            
            # Add labels and customize the plot
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(loadings_df['Indicator'], rotation=90)
            ax.set_title(f"Component {i+1} - Dominant Theme: {component_themes[f'Component_{i+1}']['dominant_theme']} "
                        f"({theme_percentages.iloc[0]}%)")
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_ylabel('Loading Value')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Themed loadings plot saved to {save_path}")
        
        # Display or close the plot
        if display_plot:
            plt.show()
        else:
            plt.close()
        
        # Create a readable summary
        print("\n==== THEMATIC COMPONENT ANALYSIS ====")
        for comp, info in component_themes.items():
            print(f"\n{comp}: Primary theme is {info['dominant_theme']}")
            for theme, percentage in info['theme_percentages'].items():
                direction = info['direction'].get(theme, 'neutral')
                print(f"  - {theme}: {percentage}% ({direction} direction)")
                
                # Show top indicators for this theme
                if theme in info['key_indicators']:
                    print("    Top indicators:")
                    for idx, ind_info in enumerate(info['key_indicators'][theme], 1):
                        print(f"    {idx}. {ind_info['Indicator']}: {ind_info['Loading']:.3f}")
        
        return component_themes

    def plot_g20_themed_biplot(self, display_plot=True, save_path=None):
        """
        Create a themed biplot specifically for G20 countries with loadings in top right

        Parameters:
        display_plot (bool): Whether to display the plot interactively
        save_path (str): Optional path to save the visualization

        Returns:
        matplotlib.figure.Figure: The figure object
        """
        if not hasattr(self, 'pca_results'):
            raise ValueError("No PCA results found. Run perform_pca first.")

        # Get the PCA results
        reduced_data = self.pca_results['reduced_data']
        loadings = self.pca_results['loadings']
        explained_variance_ratio = self.pca_results['explained_variance_ratio']

        # Create a figure with a main axis for the scatter plot
        fig = plt.figure(figsize=(14, 10))
        ax_main = plt.subplot(111)

        # Filter to only G20 countries
        g20_iso_codes = set(G20_COUNTRIES.values())
        unique_countries = [country for country in set(self.country_codes) if country in g20_iso_codes]

        # Create a colormap with distinct colors for G20 countries
        cmap = plt.cm.get_cmap('tab20', len(unique_countries))
        colors = {country: cmap(i) for i, country in enumerate(unique_countries)}

        # Create a dictionary to track which countries are plotted for the legend
        plotted_countries = {}

        # Plot each G20 country
        for country in unique_countries:
            country_mask = np.array(self.country_codes) == country
            # Skip if no data points for this country
            if not np.any(country_mask):
                continue
                
            # Get the country's full name
            country_name = ISO_TO_NAME.get(country, country)
            
            # Plot the country data points
            scatter = ax_main.scatter(
                reduced_data[country_mask, 0],
                reduced_data[country_mask, 1],
                label=country_name,
                color=colors[country],
                alpha=0.7,
                s=80
            )
            plotted_countries[country] = country_name

        # Add country and year labels to data points
        for i, (x, y, label, year) in enumerate(zip(reduced_data[:, 0], reduced_data[:, 1], 
                                                self.country_codes, self.years)):
            # Only label G20 countries
            if label in g20_iso_codes:
                if i % 3 == 0:  # Label every 3rd point to reduce clutter
                    ax_main.annotate(
                        f"{label} {year}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 5),
                        fontsize=8,
                        alpha=0.7
                    )

        # Create an inset axes for the loadings in the top right corner
        ax_inset = fig.add_axes([0.6, 0.65, 0.35, 0.3], facecolor='#f9f9f9')

        # Draw a border around the inset
        for spine in ax_inset.spines.values():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(0.5)

        # Create a DataFrame with indicator loadings and themes
        loadings_df = pd.DataFrame({
            'Indicator': self.indicators,
            'PC1': loadings[0],
            'PC2': loadings[1],
            'Theme': [INDICATOR_THEMES.get(ind, DEFAULT_THEME) for ind in self.indicators]
        })

        # Add feature vectors colored by theme to the inset axis
        for _, row in loadings_df.iterrows():
            ax_inset.arrow(
                0, 0,
                row['PC1'],
                row['PC2'],
                color=THEME_COLORS.get(row['Theme'], '#333333'),
                alpha=0.7,
                width=0.01,
                head_width=0.05,
                length_includes_head=True
            )
            
            # Add feature labels with colors matching themes
            ax_inset.annotate(
                row['Indicator'],
                (row['PC1'] * 1.1, row['PC2'] * 1.1),
                color=THEME_COLORS.get(row['Theme'], '#333333'),
                fontsize=8
            )

        # Configure the inset axis
        ax_inset.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax_inset.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax_inset.set_xlim(-1.1, 1.1)
        ax_inset.set_ylim(-1.1, 1.1)
        ax_inset.set_title("Feature Loadings by Theme", fontsize=10)
        ax_inset.set_xlabel(f"PC1", fontsize=8)
        ax_inset.set_ylabel(f"PC2", fontsize=8)
        ax_inset.grid(True, linestyle='--', alpha=0.3)

        # Add a legend for themes in the inset
        theme_legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=theme) 
                                for theme, color in THEME_COLORS.items() 
                                if theme in loadings_df['Theme'].values]

        ax_inset.legend(handles=theme_legend_elements, title="Indicator Themes", 
                        loc='best', fontsize=7, title_fontsize=8)

        # Configure the main axis
        ax_main.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}%)")
        ax_main.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}%)")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_title("G20 Countries in PCA Space with Themed Indicators", fontsize=14)

        # Create the legend for countries 
        handles, labels = ax_main.get_legend_handles_labels()

        # Place the legend outside the plot to avoid overlapping with data points
        country_legend = ax_main.legend(
            handles, labels,
            title="G20 Countries", 
            loc='center left', 
            bbox_to_anchor=(1.01, 0.5),
            fontsize=9,
            framealpha=0.9
        )
        country_legend.get_frame().set_facecolor('white')

        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"G20 themed biplot saved to {save_path}")

        # Display or close the plot
        if display_plot:
            plt.show()
        else:
            plt.close()

        return fig

# Example usage
if __name__ == "__main__":
    
    # Path to the data file - adjust as needed
    data_path = "data/all_data_merged_cleaned.json"
    
    # Initialize the dimensionality reducer
    reducer = DimensionalityReducer(data_path)
    
    # Filter years
    reducer.filter_years(1970, 2025)
    
    # Check for missing data
    reducer.check_missing_data()
    
    # Prepare for dimensionality reduction
    reducer.prepare_for_dimensionality_reduction(impute_missing=True, min_non_null_ratio=0.7)
    
    # Perform PCA
    pca_2d, explained_var, loadings = reducer.perform_pca(n_components=2)
    
    # Create and save all visualizations including themed analysis
    export_dir = 'data/dimensionality_reduction_g20'
    reducer.save_plots(export_dir)
    
    # Export thematic insights
    reducer.export_thematic_insights(export_dir)

    reducer.plot_g20_themed_biplot(display_plot=True, save_path='data/dimensionality_reduction_g20/g20_themed_biplot.png')
    
    # Export results to JSON files
    reducer.export_results(export_dir)
    
    print("Analysis complete. All results saved to:", export_dir)
    
    # Display themed analysis interactively (if running in notebook)
    # reducer.analyze_themed_loadings(n_components=2)
    # reducer.plot_themed_biplot()