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

class DimensionalityReducer:
    def __init__(self, data_path=None, data=None):
        """
        Initialize the DimensionalityReducer class
        
        Parameters:
        data_path (str): Path to the JSON data file
        data (DataFrame): Optional pre-loaded pandas DataFrame
        """
        if data is not None:
            self.df = data
        elif data_path:
            self.data = self.load_data(data_path)
            self.df = self.convert_to_dataframe(self.extract_g20_countries(self.data))
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

# Example usage
if __name__ == "__main__":
    # Path to the data file - adjust as needed
    data_path = "data/all_data_merged_cleaned.json"
    
    # Initialize the dimensionality reducer
    reducer = DimensionalityReducer(data_path)
    
    # Filter years
    reducer.filter_years(1990, 2019)
    
    # Check for missing data
    reducer.check_missing_data()
    
    # Prepare for dimensionality reduction
    reducer.prepare_for_dimensionality_reduction(impute_missing=True, min_non_null_ratio=0.7)
    
    # Perform PCA
    pca_2d, explained_var, loadings = reducer.perform_pca(n_components=2)
    
    # Perform PCA with 3 components
    pca_3d, explained_var_3d, loadings_3d = reducer.perform_pca(n_components=3)
    
    # Perform t-SNE
    tsne_data = reducer.perform_tsne(n_components=2, perplexity=30)

    # perform t-SNE with 3 components
    tsne_data_3d = reducer.perform_tsne(n_components=3, perplexity=30)
    
    # Create and save all visualizations
    export_dir = 'data/dimensionality_reduction'
    reducer.save_plots(export_dir)
    
    # Export results to JSON files
    reducer.export_results(export_dir)
    
    print("Analysis complete. All results saved to:", export_dir)
    
    # Optionally show one plot interactively at the end if running in interactive mode
    #reducer.visualize_reduced_data(technique='PCA', show_country_names=True)