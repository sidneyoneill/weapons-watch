import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryAnalyzer:
    def __init__(self, data_path):
        """
        Initialize the trajectory analyzer with data from a JSON file
        
        Parameters:
        data_path (str): Path to the t-SNE results JSON file
        """
        self.data_path = data_path
        self.df = None
        self.trajectory_features = None
        self.clusters = None
        self.cluster_labels = None
        self.min_years_required = 3  # Minimum number of years needed to compute trajectory
        
        # Load the data
        self.load_data()
        
    def load_data(self):
        """Load t-SNE data from JSON file and convert to DataFrame"""
        print(f"Loading data from {self.data_path}...")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # Store metadata
        self.metadata = data['metadata']
        
        # Convert to DataFrame
        self.df = pd.DataFrame([item for item in data['data']])
        
        # Extract coordinates into separate columns
        coords_df = pd.DataFrame(self.df['coordinates'].tolist(), 
                                index=self.df.index, 
                                columns=[f'tsne_{i+1}' for i in range(len(data['data'][0]['coordinates']))])
        self.df = pd.concat([self.df.drop('coordinates', axis=1), coords_df], axis=1)
        
        print(f"Data loaded: {len(self.df)} observations for {self.df['country_code'].nunique()} countries")
    
    def compute_trajectory_features(self):
        """
        Compute features describing each country's trajectory through t-SNE space
        """
        print("Computing trajectory features for each country...")
        
        # Group by country
        country_groups = self.df.groupby('country_code')
        
        # Initialize lists to store results
        countries = []
        feature_data = []
        
        # Get the number of t-SNE dimensions
        tsne_cols = [col for col in self.df.columns if col.startswith('tsne_')]
        n_dims = len(tsne_cols)
        
        for country_code, group in country_groups:
            # Sort by year to ensure correct trajectory computation
            group = group.sort_values('year')
            
            # Skip countries with too few observations
            if len(group) < self.min_years_required:
                print(f"Skipping {country_code}: only {len(group)} years available")
                continue
            
            # Get the country name
            country_name = group['country_name'].iloc[0]
            
            # Extract the t-SNE coordinates
            coords = group[tsne_cols].values
            
            # Compute vectors between consecutive years
            vectors = np.diff(coords, axis=0)
            
            # Compute magnitudes of each vector (distance moved each year)
            magnitudes = np.linalg.norm(vectors, axis=1)
            
            # Features to extract
            features = {
                'country_code': country_code,
                'country_name': country_name,
                'num_years': len(group),
                'start_year': group['year'].min(),
                'end_year': group['year'].max(),
                
                # Average movement vector (direction)
                'avg_vector': np.mean(vectors, axis=0),
                
                # Total distance traveled
                'total_distance': np.sum(magnitudes),
                
                # Average speed (distance per year)
                'avg_speed': np.mean(magnitudes),
                
                # Variance in speed
                'speed_variance': np.var(magnitudes),
                
                # Acceleration (increase/decrease in speed)
                'acceleration': np.mean(np.diff(magnitudes)) if len(magnitudes) > 1 else 0,
                
                # Directional consistency (how straight the path is)
                # Higher values mean more consistent direction
                'directional_consistency': np.linalg.norm(np.sum(vectors, axis=0)) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else 0,
                
                # Starting position in t-SNE space
                'start_position': coords[0],
                
                # Ending position in t-SNE space
                'end_position': coords[-1],
                
                # Overall displacement vector (end - start)
                'net_displacement': coords[-1] - coords[0],
                
                # Net displacement magnitude
                'net_displacement_magnitude': np.linalg.norm(coords[-1] - coords[0]),
                
                # Efficiency (ratio of net displacement to total distance)
                'efficiency': np.linalg.norm(coords[-1] - coords[0]) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else 0,
                
                # Store all coordinates for visualization
                'all_coordinates': coords
            }
            
            # Extract the components of the average vector for easier clustering
            for i in range(n_dims):
                features[f'avg_vector_{i+1}'] = features['avg_vector'][i]
                features[f'net_displacement_{i+1}'] = features['net_displacement'][i]
            
            countries.append(country_code)
            feature_data.append(features)
        
        # Convert to DataFrame
        self.trajectory_features = pd.DataFrame(feature_data)
        
        print(f"Computed trajectory features for {len(self.trajectory_features)} countries")
        
        return self.trajectory_features
    
    def cluster_trajectories(self, method='kmeans', n_clusters=5, features_to_use=None, scale_features=True):
        """
        Cluster countries based on their trajectory features
        
        Parameters:
        method (str): Clustering method ('kmeans', 'dbscan', or 'hierarchical')
        n_clusters (int): Number of clusters for KMeans or Hierarchical
        features_to_use (list): List of features to use for clustering; if None, use default set
        scale_features (bool): Whether to scale features before clustering
        
        Returns:
        numpy.ndarray: Cluster labels
        """
        if self.trajectory_features is None:
            raise ValueError("No trajectory features available. Run compute_trajectory_features first.")
        
        # Default features to use for clustering if not specified
        if features_to_use is None:
            # Use these trajectory features by default
            features_to_use = [
                'avg_speed', 'speed_variance', 'acceleration', 
                'directional_consistency', 'net_displacement_magnitude',
                'efficiency'
            ]
            
            # Add average vector components
            tsne_cols = [col for col in self.df.columns if col.startswith('tsne_')]
            for i in range(len(tsne_cols)):
                features_to_use.append(f'avg_vector_{i+1}')
                features_to_use.append(f'net_displacement_{i+1}')
        
        print(f"Clustering trajectories using {len(features_to_use)} features...")
        print(f"Features: {features_to_use}")
        
        # Extract features for clustering
        X = self.trajectory_features[features_to_use].values
        
        # Scale features if required
        if scale_features:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # Select clustering algorithm
        if method.lower() == 'kmeans':
            print(f"Performing K-Means clustering with {n_clusters} clusters...")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method.lower() == 'dbscan':
            print(f"Performing DBSCAN clustering...")
            # Auto-determine eps based on nearest neighbor distances
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(X)
            distances, indices = nn.kneighbors(X)
            distances = np.sort(distances[:, 1])
            eps = np.mean(distances) * 1.5  # Adjust this multiplier as needed
            
            print(f"  Auto-determined eps = {eps:.4f}")
            clusterer = DBSCAN(eps=eps, min_samples=max(5, len(X) // 20))
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
        self.trajectory_features['cluster'] = cluster_labels
        
        # Calculate silhouette score if possible
        if len(np.unique(cluster_labels)) > 1 and -1 not in cluster_labels and len(X) > len(np.unique(cluster_labels)):
            score = silhouette_score(X, cluster_labels)
            print(f"Silhouette Score: {score:.3f}")
        
        # Print cluster sizes
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        print("Cluster sizes:")
        for cluster_id, size in cluster_sizes.items():
            print(f"  Cluster {cluster_id}: {size} countries")
        
        return cluster_labels
    
    def visualize_trajectories(self, dimensions=(1, 2), save_path=None, show_labels=True, 
                             max_countries_per_plot=30, plot_clusters=True):
        """
        Visualize country trajectories in t-SNE space
        
        Parameters:
        dimensions (tuple): Which dimensions to plot
        save_path (str): Path to save the plot
        show_labels (bool): Whether to show country labels
        max_countries_per_plot (int): Maximum number of countries to show in one plot
        plot_clusters (bool): Whether to color by cluster or by country
        """
        if self.df is None:
            raise ValueError("No data available. Load data first.")
        
        # Extract the dimensions for plotting
        dim1, dim2 = dimensions
        x_col = f'tsne_{dim1}'
        y_col = f'tsne_{dim2}'
        
        if x_col not in self.df.columns or y_col not in self.df.columns:
            raise ValueError(f"Dimensions {dimensions} not found in t-SNE data")
        
        # Group by country
        country_groups = self.df.groupby('country_code')
        
        # If we're plotting by cluster, determine which countries are in which cluster
        if plot_clusters and self.cluster_labels is not None:
            clusters_df = self.trajectory_features[['country_code', 'cluster']]
            
            # Get unique clusters and colors
            unique_clusters = sorted(clusters_df['cluster'].unique())
            cmap = plt.cm.get_cmap('tab10', max(10, len(unique_clusters)))
            
            # For each cluster, plot countries in that cluster
            for cluster_id in unique_clusters:
                plt.figure(figsize=(14, 12))
                
                # Get countries in this cluster
                cluster_countries = clusters_df[clusters_df['cluster'] == cluster_id]['country_code'].values
                
                # Limit to max countries per plot
                if len(cluster_countries) > max_countries_per_plot:
                    print(f"Limiting cluster {cluster_id} plot to {max_countries_per_plot} countries")
                    cluster_countries = np.random.choice(cluster_countries, max_countries_per_plot, replace=False)
                
                # Plot each country in this cluster
                for country_code in cluster_countries:
                    group = country_groups.get_group(country_code).sort_values('year')
                    plt.plot(group[x_col], group[y_col], '-o', alpha=0.7, markersize=5)
                    
                    # Label start and end points
                    if show_labels:
                        plt.text(group[x_col].iloc[0], group[y_col].iloc[0], 
                                f"{country_code} {group['year'].iloc[0]}", fontsize=9, alpha=0.7)
                        plt.text(group[x_col].iloc[-1], group[y_col].iloc[-1], 
                                f"{country_code} {group['year'].iloc[-1]}", fontsize=9, alpha=0.7)
                
                plt.title(f'Cluster {cluster_id} Country Trajectories in t-SNE Space')
                plt.xlabel(f't-SNE Component {dim1}')
                plt.ylabel(f't-SNE Component {dim2}')
                plt.grid(True, linestyle='--', alpha=0.3)
                
                if save_path:
                    cluster_save_path = save_path.replace('.png', f'_cluster_{cluster_id}.png')
                    plt.savefig(cluster_save_path, dpi=300, bbox_inches='tight')
                    print(f"Cluster {cluster_id} plot saved to {cluster_save_path}")
                
                plt.show()
                
        else:
            # Plot all countries on one plot, colored individually
            plt.figure(figsize=(14, 12))
            
            # Get a subset of countries if there are too many
            all_countries = list(country_groups.groups.keys())
            if len(all_countries) > max_countries_per_plot:
                print(f"Limiting plot to {max_countries_per_plot} randomly selected countries")
                countries_to_plot = np.random.choice(all_countries, max_countries_per_plot, replace=False)
            else:
                countries_to_plot = all_countries
            
            # Get a color map with enough colors
            cmap = plt.cm.get_cmap('tab20', max(20, len(countries_to_plot)))
            
            # Plot each country's trajectory
            for i, country_code in enumerate(countries_to_plot):
                group = country_groups.get_group(country_code).sort_values('year')
                color = cmap(i % cmap.N)
                
                plt.plot(group[x_col], group[y_col], '-o', color=color, label=country_code, 
                        alpha=0.7, markersize=5)
                
                # Label start and end points
                if show_labels:
                    plt.text(group[x_col].iloc[0], group[y_col].iloc[0], 
                            f"{country_code} {group['year'].iloc[0]}", fontsize=9, alpha=0.7)
                    plt.text(group[x_col].iloc[-1], group[y_col].iloc[-1], 
                            f"{country_code} {group['year'].iloc[-1]}", fontsize=9, alpha=0.7)
            
            plt.title('Country Trajectories in t-SNE Space')
            plt.xlabel(f't-SNE Component {dim1}')
            plt.ylabel(f't-SNE Component {dim2}')
            plt.grid(True, linestyle='--', alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            plt.show()
    
    def visualize_cluster_features(self, save_path=None):
        """
        Visualize the distribution of trajectory features by cluster
        
        Parameters:
        save_path (str): Path to save the plot
        """
        if self.trajectory_features is None or 'cluster' not in self.trajectory_features.columns:
            raise ValueError("No clustered trajectory features available. Run compute_trajectory_features and cluster_trajectories first.")
        
        # Features to visualize
        features_to_plot = [
            'avg_speed', 'speed_variance', 'acceleration', 
            'directional_consistency', 'net_displacement_magnitude',
            'efficiency'
        ]
        
        # Create a long-format DataFrame for easier plotting with seaborn
        plot_df = pd.melt(
            self.trajectory_features, 
            id_vars=['country_code', 'cluster'], 
            value_vars=features_to_plot,
            var_name='feature', 
            value_name='value'
        )
        
        # Plot feature distributions by cluster
        plt.figure(figsize=(15, 10))
        sns.boxplot(x='feature', y='value', hue='cluster', data=plot_df)
        plt.title('Trajectory Feature Distributions by Cluster')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature plot saved to {save_path}")
        
        plt.show()
    
    def visualize_cluster_radar(self, save_path=None):
        """
        Create radar charts showing the average features for each cluster
        
        Parameters:
        save_path (str): Path to save the plot
        """
        if self.trajectory_features is None or 'cluster' not in self.trajectory_features.columns:
            raise ValueError("No clustered trajectory features available")
        
        # Features for the radar chart
        features = [
            'avg_speed', 'acceleration', 'directional_consistency',
            'net_displacement_magnitude', 'efficiency'
        ]
        
        # Calculate cluster means and scale them to 0-1 for radar chart
        cluster_means = self.trajectory_features.groupby('cluster')[features].mean()
        
        # Scale each feature to 0-1 range for better radar visualization
        for feature in features:
            min_val = cluster_means[feature].min()
            max_val = cluster_means[feature].max()
            if max_val > min_val:
                cluster_means[feature] = (cluster_means[feature] - min_val) / (max_val - min_val)
        
        # Create radar charts
        fig = plt.figure(figsize=(15, 10))
        
        # Number of variables
        num_vars = len(features)
        
        # Calculate angle for each axis
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        # Make the plot circular
        angles += angles[:1]
        
        # Add an axis for each cluster
        num_clusters = len(cluster_means)
        n_cols = 3
        n_rows = (num_clusters + n_cols - 1) // n_cols
        
        for i, (cluster_id, values) in enumerate(cluster_means.iterrows()):
            ax = fig.add_subplot(n_rows, n_cols, i+1, polar=True)
            
            # Values need to be circular
            values_list = values.tolist()
            values_list += values_list[:1]
            
            # Draw the chart
            ax.plot(angles, values_list, 'o-', linewidth=2)
            ax.fill(angles, values_list, alpha=0.25)
            
            # Set feature labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features, fontsize=8)
            
            # Configure the chart
            ax.set_yticklabels([])
            ax.set_title(f'Cluster {cluster_id} Profile', size=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Radar plot saved to {save_path}")
        
        plt.show()

    def export_cluster_results(self, export_dir, prefix='trajectory_clusters'):
        """
        Export clustering results to JSON and CSV files
        
        Parameters:
        export_dir (str): Directory to save files
        prefix (str): Prefix for filenames
        """
        if self.trajectory_features is None or 'cluster' not in self.trajectory_features.columns:
            raise ValueError("No clustered trajectory features available")
        
        os.makedirs(export_dir, exist_ok=True)
        
        # Prepare the data for export
        # For CSV, we can export the full DataFrame
        csv_path = os.path.join(export_dir, f"{prefix}.csv")
        self.trajectory_features.to_csv(csv_path, index=False)
        
        # For JSON, create a more structured format
        json_data = {
            "metadata": {
                "num_countries": len(self.trajectory_features),
                "num_clusters": len(self.trajectory_features['cluster'].unique()),
                "clustering_method": type(self.clusters).__name__ if self.clusters else "Unknown"
            },
            "clusters": []
        }
        
        # Group countries by cluster
        for cluster_id, group in self.trajectory_features.groupby('cluster'):
            cluster_info = {
                "cluster_id": int(cluster_id),
                "countries": []
            }
            
            for _, row in group.iterrows():
                country_info = {
                    "country_code": row['country_code'],
                    "country_name": row['country_name'],
                    "features": {
                        feature: float(row[feature]) if isinstance(row[feature], (int, float, np.number)) 
                                                     and not np.isnan(row[feature])
                                                     and not np.isinf(row[feature]) else None
                        for feature in ['avg_speed', 'speed_variance', 'acceleration', 
                                      'directional_consistency', 'net_displacement_magnitude',
                                      'efficiency']
                    }
                }
                cluster_info["countries"].append(country_info)
            
            json_data["clusters"].append(cluster_info)
        
        # Save to file
        json_path = os.path.join(export_dir, f"{prefix}.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Cluster results exported to {export_dir}")

    # def compute_yearly_volatility(self):
    #     """
    #     Compute the magnitude of movement vectors between consecutive years for each country
        
    #     Returns:
    #     dict: JSON-ready nested dictionary with countries, years, and volatility magnitudes
    #     """
    #     if self.df is None:
    #         raise ValueError("No data available. Load data first.")
        
    #     # Get the t-SNE dimensions
    #     tsne_cols = [col for col in self.df.columns if col.startswith('tsne_')]
        
    #     # Group by country
    #     country_groups = self.df.groupby('country_code')
        
    #     # Dictionary to store results
    #     volatility_data = {"countries": []}
        
    #     for country_code, group in country_groups:
    #         # Sort by year to ensure correct trajectory computation
    #         group = group.sort_values('year')
            
    #         # Skip countries with too few observations
    #         if len(group) < 2:  # Need at least 2 years to compute a vector
    #             print(f"Skipping {country_code}: insufficient years")
    #             continue
            
    #         # Get the country name
    #         country_name = group['country_name'].iloc[0]
            
    #         # Extract years and coordinates
    #         years = group['year'].tolist()
    #         coords = group[tsne_cols].values
            
    #         # Compute vectors between consecutive years
    #         vectors = np.diff(coords, axis=0)
            
    #         # Compute magnitudes of each vector (distance moved each year)
    #         magnitudes = np.linalg.norm(vectors, axis=1).tolist()
            
    #         # Create year-to-magnitude mapping (starting from the second year)
    #         year_volatility = []
    #         for i, year in enumerate(years[1:]):
    #             year_volatility.append({
    #                 "year": int(year),
    #                 "volatility": round(magnitudes[i], 6)  # Round to 6 decimal places
    #             })
            
    #         # Add to the result
    #         country_data = {
    #             "country_code": country_code,
    #             "country_name": country_name,
    #             "yearly_volatility": year_volatility
    #         }
    #         volatility_data["countries"].append(country_data)
        
    #     return volatility_data
    def compute_yearly_volatility(
            self,
            weights: dict = None,
            metric_weights: dict = None,
            smoothing_factor: float = 0.2,
            normalize: bool = True
        ) -> dict:
            """
            Compute volatility between consecutive years per country, combining:
              - Movement magnitude
              - Direction change (angle between trajectory segments)
              - Acceleration (change in magnitude)
            Supports:
              • Custom indicator → t‑SNE weights (or equal/default)
              • Custom metric_weights, e.g. {'magnitude':.6,'direction':.25,'acceleration':.15}
              • Exponential smoothing_factor in [0,1]
              • Normalization of each metric to [0,1]
            """
            if self.df is None or self.df.empty:
                raise ValueError("No data: load a DataFrame first.")

            # 1) identify tsne columns
            tsne_cols = [c for c in self.df.columns if c.startswith('tsne_')]
            print(f"Found {len(tsne_cols)} t-SNE columns: {tsne_cols}")
            if not tsne_cols:
                raise ValueError("No t-SNE columns found.")

            # 2) derive weights for indicators ↔ tsne dims
            indicator_weights = self._derive_indicator_weights(tsne_cols, weights)

            # 3) collect raw per-country metrics
            raw = self._collect_raw_metrics(tsne_cols)

            # 4) compute globals for normalization
            norm_params = self._compute_normalization_params(raw, normalize)

            # 5) assemble final JSON, combining metrics & smoothing
            payload = self._assemble_results(
                raw, tsne_cols, indicator_weights,
                norm_params, metric_weights or {'magnitude': .6, 'direction': .25, 'acceleration': .15},
                smoothing_factor, normalize
            )
            payload['metadata']['weights'] = indicator_weights
            return payload


    def _derive_indicator_weights(self, tsne_cols, provided_weights):
        """Either use provided, or compute from self.metadata indicators vs. each tsne dim."""
        n = len(tsne_cols)
        if provided_weights:
            return provided_weights

        indicators = self.metadata.get('indicators', [])
        df = self.df
        # if we have original indicators, correlate with each tsne dimension
        if indicators:
            w = {}
            for ind in indicators:
                if ind in df and np.issubdtype(df[ind].dtype, np.number):
                    cors = []
                    for t in tsne_cols:
                        c = np.corrcoef(df[ind], df[t])[0,1]
                        if not np.isnan(c):
                            cors.append(abs(c))
                    if cors:
                        w[ind] = np.mean(cors)
            total = sum(w.values())
            if total > 0:
                return {k: v/total for k,v in w.items()}
        # fallback: equal weights on tsne dims
        return {col: 1.0/n for col in tsne_cols}


    def _collect_raw_metrics(self, tsne_cols):
        """For each country, compute years, vectors, magnitudes, angles, accelerations."""
        groups = self.df.groupby('country_code')
        data = {}
        all_mags = []; all_angles = []; all_accs = []

        for code, g in groups:
            g = g.sort_values('year')
            if len(g) < 3:
                logging.debug(f"skip {code}: too few years")
                continue

            coords = g[tsne_cols].values
            years = list(g['year'])
            vecs = np.diff(coords, axis=0)
            mags = np.linalg.norm(vecs, axis=1)
            # angles between successive segments
            angles = []
            for i in range(len(vecs)-1):
                v1, v2 = vecs[i], vecs[i+1]
                nv1, nv2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if nv1>0 and nv2>0:
                    cos = np.clip(np.dot(v1,v2)/(nv1*nv2), -1,1)
                    angles.append(np.arccos(cos))
                else:
                    angles.append(0.0)
            accs = np.diff(mags)

            data[code] = {
                'country_name': g['country_name'].iloc[0],
                'years': years,
                'vectors':     vecs,
                'magnitudes':  mags.tolist(),
                'angles':      angles,
                'accelerations': accs.tolist()
            }
            all_mags.extend(mags)
            all_angles.extend(angles)
            all_accs.extend(accs)

        return {
            'per_country': data,
            'global': {
                'max_mag': max(all_mags)      if all_mags else 1.0,
                'max_angle': max(all_angles)  if all_angles else np.pi,
                'max_acc': max(abs(min(all_accs)), max(all_accs)) if all_accs else 1.0
            }
        }


    def _compute_normalization_params(self, raw, normalize):
        """Return tuple of divisors for mag, angle, acc—1 if not normalizing."""
        if not normalize:
            return 1.0, np.pi, 1.0
        g = raw['global']
        return g['max_mag'], g['max_angle'], g['max_acc']


    def _assemble_results(
        self, raw, tsne_cols, indicator_weights,
        norm_params, metric_weights, smoothing_factor, normalize
    ):
        max_mag, max_ang, max_acc = norm_params
        result = {'countries': [], 'metadata': {}}

        for code, info in raw['per_country'].items():
            years = info['years']
            vecs = info['vectors']
            mags = info['magnitudes']
            angles = info['angles']
            accs   = info['accelerations']

            country_entry = {
                'country_code': code,
                'country_name': info['country_name'],
                'yearly_volatility': []
            }

            prev_vol = None
            # iterate over each interval i→i+1, but report at the *arrival* year
            for i in range(len(mags)):
                norm_mag = mags[i]/max_mag if normalize else mags[i]
                dir_pen  = (angles[i-1]/max_ang) if i>0 else 0.0
                norm_acc = (accs[i-1]/max_acc) if i>0 else 0.0

                comp = (
                    metric_weights['magnitude']    * norm_mag +
                    metric_weights['direction']    * dir_pen +
                    metric_weights['acceleration'] * abs(norm_acc)
                )
                # smoothing
                if prev_vol is None:
                    smooth = comp
                else:
                    smooth = (1 - smoothing_factor)*comp + smoothing_factor*prev_vol

                prev_vol = smooth

                entry = {
                    'year': int(years[i+1]),
                    'volatility': round(smooth, 6),
                    'magnitude': round(mags[i], 6),
                    'direction_change': round(dir_pen, 6),
                    'acceleration': round(norm_acc, 6),
                    'components': {
                        tsne_cols[j]: round(vecs[i][j], 6)
                        for j in range(len(tsne_cols))
                    }
                }
                country_entry['yearly_volatility'].append(entry)

            vols = [e['volatility'] for e in country_entry['yearly_volatility']]
            country_entry['average_volatility'] = round(float(np.mean(vols)), 6)
            country_entry['max_volatility']     = round(float(np.max(vols)), 6)

            result['countries'].append(country_entry)

        return result

    def export_yearly_volatility(self, output_path, weights=None, include_direction=True, 
                            smoothing_factor=0.2, normalize=True):
        """
        Export the yearly volatility data to a JSON file
        
        Parameters:
        output_path (str): Path to save the JSON file
        weights (dict): Optional dictionary mapping indicator names to weights
        include_direction (bool): Whether to include directional components in volatility
        smoothing_factor (float): Factor to smooth volatility changes (0-1)
        normalize (bool): Whether to normalize volatility scores to 0-1 scale
        """
        volatility_data = self.compute_yearly_volatility(
            weights=weights,
            # include_direction=include_direction,
            smoothing_factor=smoothing_factor,
            normalize=normalize,
            metric_weights={'magnitude': .5, 'direction': .25, 'acceleration': .25}
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(volatility_data, f, indent=2)
        
        print(f"Yearly volatility data exported to {output_path}")
        return volatility_data

# Example usage
if __name__ == "__main__":
    # Path to the t-SNE data
    data_path = "data/clustering_results/all_countries_tsne.json"
    
    # Output directory for results
    export_dir = "data/trajectory_clusters"
    os.makedirs(export_dir, exist_ok=True)
    
    # Initialize and run the trajectory analyzer
    analyzer = TrajectoryAnalyzer(data_path)
    
    # Compute trajectory features
    trajectory_features = analyzer.compute_trajectory_features()
    
    # Display some statistics about the trajectories
    print("\nTrajectory Feature Statistics:")
    stats = trajectory_features[['avg_speed', 'directional_consistency', 
                               'net_displacement_magnitude', 'efficiency']].describe()
    print(stats)
    
    '''
    # Cluster the trajectories with different methods
    analyzer.cluster_trajectories(method='kmeans', n_clusters=5)
    
    
    # Visualize trajectories by cluster
    analyzer.visualize_trajectories(
        dimensions=(1, 2),
        save_path=os.path.join(export_dir, "trajectory_clusters_dim1_2.png"),
        plot_clusters=True,
        max_countries_per_plot=15
    )
    
    # Visualize trajectories for dimensions 1,3 and 2,3
    analyzer.visualize_trajectories(
        dimensions=(1, 3),
        save_path=os.path.join(export_dir, "trajectory_clusters_dim1_3.png"),
        plot_clusters=True,
        max_countries_per_plot=15
    )
    
    analyzer.visualize_trajectories(
        dimensions=(2, 3),
        save_path=os.path.join(export_dir, "trajectory_clusters_dim2_3.png"),
        plot_clusters=True,
        max_countries_per_plot=15
    )
    
    
    # Visualize feature distributions by cluster
    analyzer.visualize_cluster_features(
        save_path=os.path.join(export_dir, "cluster_feature_distributions.png")
    )
    
    # Create radar charts for cluster profiles
    analyzer.visualize_cluster_radar(
        save_path=os.path.join(export_dir, "cluster_radar_profiles.png")
    )
    '''
    # Example with custom weights
    weights = {
        # 'tsne_0': 0.1,
        'tsne_1': 0.4,
        'tsne_2': 0.5,
        'tsne_3': 0.1,
        # 'tsne_4': 0.15
    }
    # weights = None

    volatility_data = analyzer.export_yearly_volatility(
        output_path=os.path.join(export_dir, "country_yearly_volatility_weighted.json"),
        weights=weights,
        smoothing_factor=0.3,
        normalize=False
    )

    # Export cluster results
    analyzer.export_cluster_results(export_dir)
    
    print("Trajectory analysis and clustering complete!")