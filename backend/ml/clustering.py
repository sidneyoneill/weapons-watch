import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Import the DimensionalityReducer class
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dimensionality_reduction import DimensionalityReducer

class CountryClustering:
    def __init__(self, data_path=None, data=None, time_range=None):
        """
        Initialize the CountryClustering class
        
        Parameters:
        data_path (str): Path to the JSON data file
        data (DataFrame): Optional pre-loaded pandas DataFrame
        time_range (tuple): Optional (start_year, end_year) to filter data
        """
        self.data_path = data_path
        self.reducer = None
        self.tsne_data = None
        self.clusters = None
        self.cluster_labels = None
        self.time_range = time_range if time_range else (None, None)
        
        # Will be populated during processing
        self.country_codes = []
        self.years = []
        self.row_labels = []
        
    def process_all_countries(self, n_components=3, perplexity=30, n_iter=1000, 
                             min_non_null_ratio=0.5, export_dir=None):
        """
        Process all countries using t-SNE for dimensionality reduction
        
        Parameters:
        n_components (int): Number of t-SNE components
        perplexity (float): Perplexity parameter for t-SNE
        n_iter (int): Number of iterations for t-SNE optimization
        min_non_null_ratio (float): Minimum ratio of non-null values needed for an indicator
        export_dir (str): Directory to export results (if None, no export)
        
        Returns:
        pd.DataFrame: DataFrame containing t-SNE coordinates and metadata
        """
        print("Loading data and preparing for t-SNE...")
        
        # Initialize the DimensionalityReducer with all countries by setting g20_only=False
        self.reducer = DimensionalityReducer(data_path=self.data_path, g20_only=False)
        
        # Filter by year range if provided
        if self.time_range[0] and self.time_range[1]:
            self.reducer.filter_years(self.time_range[0], self.time_range[1])
            print(f"Filtered years to range: {self.time_range[0]}-{self.time_range[1]}")
        
        # Check for missing data
        self.reducer.check_missing_data()
        
        # Prepare data for dimensionality reduction
        # The key difference: we're using the DimensionalityReducer but not limiting to G20 countries
        # By not providing a countries parameter, we include all countries
        self.reducer.prepare_for_dimensionality_reduction(
            impute_missing=True, 
            min_non_null_ratio=min_non_null_ratio
        )
        
        # Perform t-SNE with specified components
        print(f"Performing t-SNE with {n_components} dimensions...")
        tsne_result = self.reducer.perform_tsne(
            n_components=n_components, 
            perplexity=perplexity, 
            n_iter=n_iter
        )
        
        # Store metadata from reducer for later use
        self.country_codes = self.reducer.country_codes
        self.years = self.reducer.years
        self.row_labels = self.reducer.row_labels
        
        # Create a DataFrame with the results
        results_df = self.create_results_dataframe(tsne_result)
        self.tsne_data = results_df
        
        # Export if a directory is provided
        if export_dir:
            self.export_tsne_results(export_dir)
        
        print(f"t-SNE processing complete. Data contains {len(results_df)} country-year observations.")
        return results_df
    
    def create_results_dataframe(self, tsne_result):
        """
        Create a DataFrame with t-SNE results and metadata
        
        Parameters:
        tsne_result (numpy.ndarray): The t-SNE coordinates
        
        Returns:
        pd.DataFrame: DataFrame with t-SNE coordinates and metadata
        """
        # Base columns for the DataFrame
        data = {
            'country_code': self.country_codes,
            'year': self.years,
            'label': self.row_labels
        }
        
        # Add t-SNE coordinates
        for i in range(tsne_result.shape[1]):
            data[f'tsne_{i+1}'] = tsne_result[:, i]
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # For non-G20 countries, ISO_TO_NAME won't have a mapping, so use the ISO code itself
        from dimensionality_reduction import ISO_TO_NAME
        df['country_name'] = df['country_code'].map(lambda x: ISO_TO_NAME.get(x, x))
        
        return df
    
    def perform_clustering(self, method='kmeans', n_clusters=5, eps=0.5, min_samples=5):
        """
        Perform clustering on the t-SNE reduced data
        
        Parameters:
        method (str): Clustering method ('kmeans', 'dbscan', or 'hierarchical')
        n_clusters (int): Number of clusters for KMeans or Hierarchical
        eps (float): Maximum distance between samples for DBSCAN
        min_samples (int): Minimum samples in neighborhood for DBSCAN
        
        Returns:
        numpy.ndarray: Cluster labels
        """
        if self.tsne_data is None:
            raise ValueError("No t-SNE data available. Run process_all_countries first.")
        
        # Extract t-SNE coordinates for clustering
        tsne_cols = [col for col in self.tsne_data.columns if col.startswith('tsne_')]
        X = self.tsne_data[tsne_cols].values
        
        # Select clustering algorithm
        if method.lower() == 'kmeans':
            print(f"Performing K-Means clustering with {n_clusters} clusters...")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method.lower() == 'dbscan':
            print(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        elif method.lower() == 'hierarchical':
            print(f"Performing Hierarchical clustering with {n_clusters} clusters...")
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError("Unsupported clustering method. Use 'kmeans', 'dbscan', or 'hierarchical'.")
        
        # Perform clustering
        cluster_labels = clusterer.fit_predict(X)
        self.clusters = clusterer
        self.cluster_labels = cluster_labels
        
        # Add cluster labels to the DataFrame
        self.tsne_data['cluster'] = cluster_labels
        
        # Calculate silhouette score if possible
        if len(np.unique(cluster_labels)) > 1 and -1 not in cluster_labels:
            score = silhouette_score(X, cluster_labels)
            print(f"Silhouette Score: {score:.3f}")
        
        return cluster_labels
    
    def visualize_clusters(self, dimensions=(1, 2), colors=None, markers=None, save_path=None):
        """
        Visualize the clusters in 2D
        
        Parameters:
        dimensions (tuple): Which dimensions to plot (e.g., (1, 2) for 1st and 2nd t-SNE components)
        colors (list): Optional list of colors for clusters
        markers (list): Optional list of markers for countries/regions
        save_path (str): Optional path to save the plot
        """
        if self.tsne_data is None or 'cluster' not in self.tsne_data.columns:
            raise ValueError("No clustering results available. Run perform_clustering first.")
        
        # Extract dimensions for plotting
        dim1, dim2 = dimensions
        x_col = f'tsne_{dim1}'
        y_col = f'tsne_{dim2}'
        
        if x_col not in self.tsne_data.columns or y_col not in self.tsne_data.columns:
            raise ValueError(f"Dimensions {dimensions} not found in t-SNE data.")
        
        # Set up the plot
        plt.figure(figsize=(14, 10))
        
        # Get unique cluster labels
        unique_clusters = sorted(self.tsne_data['cluster'].unique())
        
        # Set default colors if not provided
        if colors is None:
            cmap = plt.cm.get_cmap('tab10', len(unique_clusters))
            colors = [cmap(i) for i in range(len(unique_clusters))]
        
        # Plot each cluster
        for i, cluster in enumerate(unique_clusters):
            cluster_data = self.tsne_data[self.tsne_data['cluster'] == cluster]
            
            # Choose a color for this cluster
            color = colors[i % len(colors)]
            
            # Plot the points
            plt.scatter(
                cluster_data[x_col], 
                cluster_data[y_col],
                s=80, 
                c=[color], 
                label=f'Cluster {cluster}',
                alpha=0.7
            )
        
        # Add labels for a few representative countries in each cluster
        for cluster in unique_clusters:
            cluster_data = self.tsne_data[self.tsne_data['cluster'] == cluster]
            
            # If cluster has countries, add some labels
            if len(cluster_data) > 0:
                # Group by country and get the most recent year for each
                countries = cluster_data.groupby('country_name').apply(
                    lambda x: x.loc[x['year'].idxmax()]
                )
                
                # Label a few countries (adjust number as needed)
                for _, country in countries.iloc[:5].iterrows():
                    plt.annotate(
                        country['country_name'],
                        (country[x_col], country[y_col]),
                        fontsize=9,
                        alpha=0.8,
                        xytext=(5, 5),
                        textcoords='offset points'
                    )
        
        plt.title(f'Country Clusters (t-SNE dimensions {dim1} and {dim2})')
        plt.xlabel(f't-SNE Component {dim1}')
        plt.ylabel(f't-SNE Component {dim2}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster visualization saved to {save_path}")
        
        plt.show()
    
    def export_tsne_results(self, export_dir, filename='all_countries_tsne.json'):
        """
        Export t-SNE results to a JSON file
        
        Parameters:
        export_dir (str): Directory to save the file
        filename (str): Name of the file
        """
        if self.tsne_data is None:
            raise ValueError("No t-SNE data available. Run process_all_countries first.")
        
        os.makedirs(export_dir, exist_ok=True)
        
        # Prepare the data for export
        export_data = {
            'metadata': {
                'indicators_used': self.reducer.indicators,
                'num_countries': len(self.tsne_data['country_code'].unique()),
                'num_years': len(self.tsne_data['year'].unique()),
                'total_observations': len(self.tsne_data)
            },
            'data': []
        }
        
        # Add cluster information if available
        if 'cluster' in self.tsne_data.columns:
            export_data['metadata']['num_clusters'] = len(self.tsne_data['cluster'].unique())
        
        # Convert each row to a dictionary for the JSON export
        for _, row in self.tsne_data.iterrows():
            entry = {
                'country_code': row['country_code'],
                'country_name': row['country_name'],
                'year': int(row['year']),
                'coordinates': [row[f'tsne_{i+1}'] for i in range(len([c for c in self.tsne_data.columns if c.startswith('tsne_')]))]
            }
            
            # Add cluster if available
            if 'cluster' in self.tsne_data.columns:
                entry['cluster'] = int(row['cluster'])
            
            export_data['data'].append(entry)
        
        # Save to file
        filepath = os.path.join(export_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"t-SNE results exported to {filepath}")
        
        # Also save as CSV for easier analysis
        csv_filepath = os.path.join(export_dir, filename.replace('.json', '.csv'))
        self.tsne_data.to_csv(csv_filepath, index=False)
        print(f"t-SNE results also exported as CSV to {csv_filepath}")


# Example usage
if __name__ == "__main__":
    # Path to the data file - adjust as needed
    data_path = "data/all_data_merged_cleaned.json"
    
    # Define output directory
    export_dir = "data/clustering_results"
    
    # Initialize clustering with time range
    clustering = CountryClustering(data_path=data_path, time_range=(1990, 2019))
    
    # Process all countries with t-SNE (3 dimensions)
    tsne_data = clustering.process_all_countries(
        n_components=3,
        perplexity=40,  # Higher perplexity for more countries
        n_iter=2000,    # More iterations for stability
        min_non_null_ratio=0.5,  # Lower threshold to include more indicators
        export_dir=export_dir
    )
    
    # Perform clustering
    clustering.perform_clustering(method='kmeans', n_clusters=5)
    
    # Visualize different 2D projections of the 3D t-SNE space
    clustering.visualize_clusters(
        dimensions=(1, 2), 
        save_path=os.path.join(export_dir, "tsne_clusters_dim1_2.png")
    )
    
    clustering.visualize_clusters(
        dimensions=(1, 3), 
        save_path=os.path.join(export_dir, "tsne_clusters_dim1_3.png")
    )
    
    clustering.visualize_clusters(
        dimensions=(2, 3), 
        save_path=os.path.join(export_dir, "tsne_clusters_dim2_3.png")
    )
    
    print("Clustering analysis complete.")