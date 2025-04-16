import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from collections import defaultdict

class VolatilityAnalyzer:
    """
    Analyzes and visualizes volatility data from t-SNE trajectories
    """
    
    def __init__(self, data_path=None, data=None):
        """
        Initialize the volatility analyzer
        
        Parameters:
        data_path (str): Path to the volatility JSON file
        data (dict): Pre-loaded volatility data dictionary
        """
        self.data = None
        self.df = None
        
        if data is not None:
            self.data = data
            self._process_data()
        elif data_path:
            self.load_data(data_path)
        else:
            print("No data provided. Use load_data() to load volatility data.")
    
    def load_data(self, data_path):
        """
        Load volatility data from JSON file
        
        Parameters:
        data_path (str): Path to the volatility JSON file
        """
        print(f"Loading volatility data from {data_path}...")
        
        try:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
            
            self._process_data()
            print(f"Data loaded: volatility information for {len(self.data['countries'])} countries")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _process_data(self):
        """Convert nested JSON data to a flat DataFrame for easier analysis"""
        rows = []
        
        for country in self.data['countries']:
            country_code = country['country_code']
            country_name = country['country_name']
            
            for year_data in country['yearly_volatility']:
                rows.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'year': year_data['year'],
                    'volatility': year_data['volatility']
                })
        
        self.df = pd.DataFrame(rows)
        
        # Sort by country and year
        self.df = self.df.sort_values(['country_code', 'year'])
        
        # Calculate global statistics per year
        self.yearly_stats = self.df.groupby('year').agg({
            'volatility': ['mean', 'median', 'std', 'min', 'max', 'count']
        }).reset_index()
        
        # Flatten the column names
        self.yearly_stats.columns = ['year', 'mean_volatility', 'median_volatility', 
                                     'std_volatility', 'min_volatility', 'max_volatility', 
                                     'country_count']
    
    def get_country_volatility(self, country_code):
        """
        Get volatility data for a specific country
        
        Parameters:
        country_code (str): ISO country code
        
        Returns:
        DataFrame: Volatility data for the specified country
        """
        if self.df is None:
            raise ValueError("No data available. Load data first.")
        
        country_data = self.df[self.df['country_code'] == country_code]
        
        if country_data.empty:
            raise ValueError(f"No data found for country code '{country_code}'")
        
        return country_data
    
    def visualize_country_volatility(self, country_code, comparison='global_avg', 
                                   highlight_events=None, save_path=None, display_plot=True):
        """
        Visualize volatility for a specific country over time
        
        Parameters:
        country_code (str): ISO country code
        comparison (str): What to compare against ('global_avg', 'global_median', 'none')
        highlight_events (dict): Dictionary mapping years to event labels
        save_path (str): Path to save the visualization
        display_plot (bool): Whether to display the plot
        
        Returns:
        Figure: Matplotlib figure object
        """
        try:
            country_data = self.get_country_volatility(country_code)
            country_name = country_data['country_name'].iloc[0]
            
            fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
            
            # Plot country volatility
            ax.plot(country_data['year'], country_data['volatility'], 
                  marker='o', linestyle='-', linewidth=2, 
                  color='#1f77b4', label=f"{country_name} Volatility")
            
            # Add comparison if requested
            if comparison == 'global_avg':
                ax.plot(self.yearly_stats['year'], self.yearly_stats['mean_volatility'], 
                      linestyle='--', color='#ff7f0e', alpha=0.7, 
                      label='Global Average Volatility')
            elif comparison == 'global_median':
                ax.plot(self.yearly_stats['year'], self.yearly_stats['median_volatility'], 
                      linestyle='--', color='#ff7f0e', alpha=0.7, 
                      label='Global Median Volatility')
            
            # Highlight significant events if provided
            if highlight_events:
                for year, event in highlight_events.items():
                    year_data = country_data[country_data['year'] == year]
                    if not year_data.empty:
                        ax.axvline(x=year, color='red', alpha=0.3, linestyle=':')
                        ax.annotate(event, 
                                  xy=(year, year_data['volatility'].iloc[0]),
                                  xytext=(10, 20), 
                                  textcoords='offset points',
                                  arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
            
            # Set plot title and labels
            ax.set_title(f'Volatility Index for {country_name} ({country_code})', fontsize=14)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Volatility (magnitude of movement in t-SNE space)', fontsize=12)
            
            # Configure axis
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Add legend
            ax.legend(loc='best')
            
            # Annotate average volatility
            avg_volatility = country_data['volatility'].mean()
            ax.axhline(y=avg_volatility, color='blue', alpha=0.3, linestyle='-.')
            ax.annotate(f'Average: {avg_volatility:.4f}', 
                      xy=(country_data['year'].min(), avg_volatility),
                      xytext=(10, 10), 
                      textcoords='offset points')
            
            plt.tight_layout()
            
            # Save the plot if a path is provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            # Display or close the plot
            if display_plot:
                plt.show()
            else:
                plt.close()
            
            return fig
            
        except Exception as e:
            print(f"Error visualizing country volatility: {e}")
            return None
    
    def visualize_global_volatility(self, metric='mean', include_std=True, 
                                  top_volatile_countries=5, save_path=None, display_plot=True):
        """
        Visualize global volatility trends over time
        
        Parameters:
        metric (str): Which metric to visualize ('mean', 'median')
        include_std (bool): Whether to include standard deviation band
        top_volatile_countries (int): Number of most volatile countries to highlight
        save_path (str): Path to save the visualization
        display_plot (bool): Whether to display the plot
        
        Returns:
        Figure: Matplotlib figure object
        """
        if self.df is None or self.yearly_stats is None:
            raise ValueError("No data available. Load data first.")
        
        fig, ax = plt.figure(figsize=(14, 8)), plt.gca()
        
        # Plot the main volatility trend
        if metric == 'mean':
            ax.plot(self.yearly_stats['year'], self.yearly_stats['mean_volatility'], 
                  marker='o', linestyle='-', linewidth=2, 
                  color='#1f77b4', label='Global Mean Volatility')
            
            # Add standard deviation band if requested
            if include_std:
                upper = self.yearly_stats['mean_volatility'] + self.yearly_stats['std_volatility']
                lower = self.yearly_stats['mean_volatility'] - self.yearly_stats['std_volatility']
                lower = np.maximum(lower, 0)  # Ensure lower bound is not negative
                
                ax.fill_between(self.yearly_stats['year'], lower, upper, 
                              alpha=0.2, color='#1f77b4', 
                              label='±1 Standard Deviation')
        
        elif metric == 'median':
            ax.plot(self.yearly_stats['year'], self.yearly_stats['median_volatility'], 
                  marker='o', linestyle='-', linewidth=2, 
                  color='#1f77b4', label='Global Median Volatility')
        
        # Highlight years with highest volatility
        peak_years = self.yearly_stats.nlargest(3, f'{metric}_volatility')
        for _, row in peak_years.iterrows():
            ax.annotate(f"Peak: {row['year']}", 
                      xy=(row['year'], row[f'{metric}_volatility']),
                      xytext=(0, 20), 
                      textcoords='offset points',
                      arrowprops=dict(arrowstyle='->', color='red'),
                      ha='center')
        
        # Identify top volatile countries
        if top_volatile_countries > 0:
            country_avg_volatility = self.df.groupby(['country_code', 'country_name'])['volatility'].mean().reset_index()
            top_countries = country_avg_volatility.nlargest(top_volatile_countries, 'volatility')
            
            # Get data for top volatile countries
            for _, row in top_countries.iterrows():
                country_data = self.df[self.df['country_code'] == row['country_code']]
                #ax.plot(country_data['year'], country_data['volatility'], 
                #     linestyle='--', alpha=0.7, 
                 #     label=f"{row['country_name']} ({row['country_code']})")
        
        # Set plot title and labels
        ax.set_title(f'Global Volatility Index Over Time', fontsize=14)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Volatility (magnitude of movement in t-SNE space)', fontsize=12)
        
        # Configure axis
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legend
        ax.legend(loc='best')
        
        # Add context information
        ax.text(0.02, 0.02, 
              f"Based on data from {len(self.data['countries'])} countries\n"
              f"Years: {self.yearly_stats['year'].min()} to {self.yearly_stats['year'].max()}",
              transform=ax.transAxes, fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Display or close the plot
        if display_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def compare_countries(self, country_codes, save_path=None, display_plot=True):
        """
        Compare volatility patterns between multiple countries
        
        Parameters:
        country_codes (list): List of ISO country codes to compare
        save_path (str): Path to save the visualization
        display_plot (bool): Whether to display the plot
        
        Returns:
        Figure: Matplotlib figure object
        """
        if self.df is None:
            raise ValueError("No data available. Load data first.")
        
        fig, ax = plt.figure(figsize=(14, 8)), plt.gca()
        
        # Plot global average for reference
        ax.plot(self.yearly_stats['year'], self.yearly_stats['mean_volatility'], 
              linestyle='--', color='black', alpha=0.5, 
              label='Global Average')
        
        # Plot each country
        for country_code in country_codes:
            try:
                country_data = self.get_country_volatility(country_code)
                country_name = country_data['country_name'].iloc[0]
                
                ax.plot(country_data['year'], country_data['volatility'], 
                      marker='o', linestyle='-', 
                      label=f"{country_name} ({country_code})")
            except ValueError as e:
                print(f"Warning: {e}")
        
        # Set plot title and labels
        ax.set_title(f'Volatility Comparison Between Countries', fontsize=14)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Volatility (magnitude of movement in t-SNE space)', fontsize=12)
        
        # Configure axis
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legend
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Display or close the plot
        if display_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def identify_volatility_events(self, threshold=1.5, min_countries=3):
        """
        Identify years with unusually high global volatility
        
        Parameters:
        threshold (float): How many standard deviations above the mean to consider 'high'
        min_countries (int): Minimum number of countries that must show high volatility
        
        Returns:
        DataFrame: Years with high volatility and affected countries
        """
        if self.df is None:
            raise ValueError("No data available. Load data first.")
        
        # Calculate the overall mean and standard deviation of volatility
        overall_mean = self.df['volatility'].mean()
        overall_std = self.df['volatility'].std()
        
        # Define the threshold for 'high' volatility
        high_threshold = overall_mean + (threshold * overall_std)
        
        # Find instances of high volatility
        high_volatility = self.df[self.df['volatility'] > high_threshold].copy()
        
        # Group by year and count countries
        year_counts = high_volatility.groupby('year').size().reset_index(name='country_count')
        
        # Filter for years with at least min_countries countries showing high volatility
        significant_years = year_counts[year_counts['country_count'] >= min_countries]
        
        if significant_years.empty:
            print(f"No significant volatility events found with threshold={threshold} and min_countries={min_countries}")
            return pd.DataFrame()
        
        # Create a detailed result with countries affected
        results = []
        
        for _, row in significant_years.iterrows():
            year = row['year']
            year_data = high_volatility[high_volatility['year'] == year]
            
            # Get countries with high volatility this year
            countries = [f"{row['country_name']} ({row['country_code']}): {row['volatility']:.4f}" 
                        for _, row in year_data.iterrows()]
            
            # Add to results
            results.append({
                'year': year,
                'country_count': row['country_count'],
                'affected_countries': countries,
                'mean_volatility': year_data['volatility'].mean(),
                'global_mean_that_year': self.yearly_stats[self.yearly_stats['year'] == year]['mean_volatility'].iloc[0]
            })
        
        return pd.DataFrame(results)
    
    def export_volatility_data(self, export_dir, prefix='volatility_analysis'):
        """
        Export volatility analysis data to CSV and JSON files
        
        Parameters:
        export_dir (str): Directory to save files
        prefix (str): Prefix for filenames
        """
        if self.df is None:
            raise ValueError("No data available. Load data first.")
        
        os.makedirs(export_dir, exist_ok=True)
        
        # Export full data to CSV
        csv_path = os.path.join(export_dir, f"{prefix}_data.csv")
        self.df.to_csv(csv_path, index=False)
        
        # Export yearly statistics to CSV
        stats_path = os.path.join(export_dir, f"{prefix}_yearly_stats.csv")
        self.yearly_stats.to_csv(stats_path, index=False)
        
        # Export country rankings by average volatility
        country_avg = self.df.groupby(['country_code', 'country_name'])['volatility'].agg(['mean', 'std', 'min', 'max']).reset_index()
        country_avg = country_avg.sort_values('mean', ascending=False)
        rankings_path = os.path.join(export_dir, f"{prefix}_country_rankings.csv")
        country_avg.to_csv(rankings_path, index=False)
        
        # Identify volatility events
        try:
            events = self.identify_volatility_events()
            if not events.empty:
                events_path = os.path.join(export_dir, f"{prefix}_volatility_events.csv")
                events.to_csv(events_path, index=False)
        except Exception as e:
            print(f"Warning: Could not generate volatility events: {e}")
        
        print(f"Volatility analysis data exported to {export_dir}")

# Example usage
if __name__ == "__main__":
    # Path to the volatility data
    data_path = "data/trajectory_clusters/country_yearly_volatility.json"
    
    # Output directory for results
    export_dir = "../data/volatility_analysis"
    os.makedirs(export_dir, exist_ok=True)
    
    # Initialize the volatility analyzer
    analyzer = VolatilityAnalyzer(data_path)
    
    # Visualize global volatility trends
    analyzer.visualize_global_volatility(
        save_path=os.path.join(export_dir, "global_volatility_trend.png")
    )
    
    # Visualize individual country volatility
    for country in ["USA", "RUS", "CHN", "DEU", "BRA"]:
        try:
            analyzer.visualize_country_volatility(
                country_code=country,
                comparison='global_avg',
                save_path=os.path.join(export_dir, f"{country}_volatility.png")
            )
        except ValueError:
            print(f"Skipping {country}: no data available")
    
    # Compare major powers
    analyzer.compare_countries(
        country_codes=["USA", "RUS", "CHN", "GBR", "FRA"],
        save_path=os.path.join(export_dir, "major_powers_comparison.png")
    )
    
    # Identify and print significant volatility events
    events = analyzer.identify_volatility_events(threshold=1.5, min_countries=3)
    if not events.empty:
        print("\nSignificant Volatility Events:")
        for _, event in events.iterrows():
            print(f"Year {event['year']}: {event['country_count']} countries with high volatility")
            print(f"  Mean volatility: {event['mean_volatility']:.4f} (global mean that year: {event['global_mean_that_year']:.4f})")
            print(f"  Affected countries (showing top 5):")
            for country in event['affected_countries'][:5]:
                print(f"    {country}")
            print()
    
    # Export data
    analyzer.export_volatility_data(export_dir)
    
    print("Volatility analysis complete!")