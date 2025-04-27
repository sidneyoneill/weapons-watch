import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def load_volatility_data(json_path):
    """Load volatility data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def convert_to_dataframe(volatility_data):
    """Convert JSON structure to pandas DataFrame for easier analysis"""
    rows = []
    
    for country in volatility_data['countries']:
        country_code = country['country_code']
        country_name = country['country_name']
        avg_volatility = country['average_volatility']
        max_volatility = country['max_volatility']
        
        for yearly_data in country['yearly_volatility']:
            row = {
                'country_code': country_code,
                'country_name': country_name,
                'year': yearly_data['year'],
                'volatility': yearly_data['volatility'],
                'magnitude': yearly_data['magnitude'],
                'direction_change': yearly_data['direction_change'],
                'acceleration': yearly_data['acceleration'],
                'avg_volatility': avg_volatility,
                'max_volatility': max_volatility
            }
            
            # Add component data if available
            if 'components' in yearly_data:
                for comp_name, comp_value in yearly_data['components'].items():
                    row[comp_name] = comp_value
                    
            rows.append(row)
    
    return pd.DataFrame(rows)

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Visualizations will be saved to: {output_dir}")

def plot_top_volatile_countries(df, output_dir, n=20):
    """Plot top N most volatile countries based on average volatility"""
    plt.figure(figsize=(12, 8))
    
    # Get top N countries by average volatility
    top_countries = df.groupby('country_code')['avg_volatility'].first().nlargest(n)
    
    # Create a bar plot
    ax = sns.barplot(x=top_countries.index, y=top_countries.values, palette='viridis')
    
    # Add country names as labels (use first letter if name is too long)
    country_labels = []
    for code in top_countries.index:
        name = df[df['country_code'] == code]['country_name'].iloc[0]
        if name == code or len(name) <= 10:
            country_labels.append(name)
        else:
            country_labels.append(f"{code}\n({name[:7]}...)")
    
    plt.xticks(range(len(top_countries)), country_labels, rotation=45, ha='right')
    
    plt.title(f'Top {n} Most Volatile Countries (Avg. Volatility)')
    plt.xlabel('Country')
    plt.ylabel('Average Volatility Score')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'top_{n}_volatile_countries.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_global_volatility_over_time(df, output_dir):
    """Plot global average volatility trend over time"""
    plt.figure(figsize=(14, 8))
    
    # Calculate average volatility across all countries by year
    yearly_avg = df.groupby('year')['volatility'].mean().reset_index()
    yearly_std = df.groupby('year')['volatility'].std().reset_index()
    
    # Plot line with confidence interval
    plt.plot(yearly_avg['year'], yearly_avg['volatility'], marker='o', linewidth=2, color='#1f77b4')
    plt.fill_between(
        yearly_avg['year'],
        yearly_avg['volatility'] - yearly_std['volatility'],
        yearly_avg['volatility'] + yearly_std['volatility'],
        alpha=0.3, color='#1f77b4'
    )
    
    # Highlight significant events (you can customize these)
    events = {
        2001: "9/11 Attacks",
        2008: "Financial Crisis",
        2011: "Arab Spring",
        2014: "Crimea Annexation",
        2020: "COVID-19 Pandemic"
    }
    
    for year, label in events.items():
        if year in yearly_avg['year'].values:
            idx = yearly_avg[yearly_avg['year'] == year].index[0]
            plt.axvline(x=year, color='red', linestyle='--', alpha=0.7)
            plt.text(year + 0.2, 0.9 * yearly_avg['volatility'].max(), label, 
                     rotation=90, va='top', fontsize=9)
    
    plt.title('Global Average Volatility Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Volatility Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(yearly_avg['year'][::2])  # Show every other year
    
    plt.savefig(os.path.join(output_dir, 'global_volatility_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_volatility_components_heatmap(df, output_dir):
    """Plot heatmap showing relationship between volatility components"""
    plt.figure(figsize=(10, 8))
    
    # Select columns for correlation
    cols_to_correlate = ['volatility', 'magnitude', 'direction_change', 'acceleration']
    
    # Find columns that start with 'tsne_'
    tsne_cols = [col for col in df.columns if col.startswith('tsne_')]
    cols_to_correlate.extend(tsne_cols)
    
    # Compute correlation matrix
    corr = df[cols_to_correlate].corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    plt.title('Correlation Between Volatility Components')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'volatility_components_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_country_comparison(df, output_dir, countries=None):
    """Compare volatility patterns between selected countries"""
    if not countries:
        # If no countries specified, take 5 with highest average volatility
        # and 5 with lowest for comparison
        top_countries = df.groupby('country_code')['avg_volatility'].mean().nlargest(5).index.tolist()
        bottom_countries = df.groupby('country_code')['avg_volatility'].mean().nsmallest(5).index.tolist()
        countries = top_countries + bottom_countries
    
    plt.figure(figsize=(14, 10))
    
    # Filter data for specified countries
    filtered_df = df[df['country_code'].isin(countries)]
    
    # Create plot for each country with labels
    sns.lineplot(
        data=filtered_df, 
        x='year', 
        y='volatility', 
        hue='country_code',
        style='country_code',
        markers=True, 
        dashes=False,
        linewidth=2
    )
    
    # Add country name tooltips to the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    country_names = {}
    for code in countries:
        name = df[df['country_code'] == code]['country_name'].iloc[0]
        country_names[code] = f"{code} ({name})"
    
    new_labels = [country_names.get(label, label) for label in labels]
    plt.legend(handles, new_labels, title='Country', loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title('Country Volatility Comparison')
    plt.xlabel('Year')
    plt.ylabel('Volatility Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'country_volatility_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_volatility_heatmap(df, output_dir, n_countries=30):
    """Create a heatmap showing volatility across years for top N volatile countries"""
    plt.figure(figsize=(16, 12))
    
    # Get top N countries by average volatility
    top_countries = df.groupby('country_code')['avg_volatility'].first().nlargest(n_countries).index.tolist()
    
    # Filter data for these countries
    filtered_df = df[df['country_code'].isin(top_countries)]
    
    # Pivot data for heatmap
    pivot_data = filtered_df.pivot_table(
        index='country_code', 
        columns='year', 
        values='volatility',
        aggfunc='mean'
    )
    
    # Sort by average volatility
    pivot_data = pivot_data.reindex(top_countries)
    
    # Create a custom colormap from green to yellow to red
    colors = ['#2c7bb6', '#ffffbf', '#d7191c']  # Blue to yellow to red
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
    
    # Create heatmap
    ax = sns.heatmap(
        pivot_data, 
        cmap=cmap,
        linewidths=0.5, 
        linecolor='lightgray',
        cbar_kws={'label': 'Volatility Score'}
    )
    
    # Add country names next to codes
    country_labels = []
    for code in top_countries:
        name = df[df['country_code'] == code]['country_name'].iloc[0]
        if name == code or len(name) > 15:
            country_labels.append(code)
        else:
            country_labels.append(f"{code} ({name})")
    
    plt.yticks(np.arange(len(top_countries)) + 0.5, country_labels, rotation=0)
    
    plt.title(f'Volatility Heatmap for Top {n_countries} Most Volatile Countries')
    plt.xlabel('Year')
    plt.ylabel('Country')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'volatility_heatmap_top{n_countries}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_component_breakdown(df, output_dir, country_code=None):
    """Plot the breakdown of volatility components for a specific country"""
    if country_code is None:
        # If no country specified, pick the one with the highest volatility
        country_code = df.groupby('country_code')['avg_volatility'].mean().idxmax()
    
    # Filter data for the specific country
    country_data = df[df['country_code'] == country_code]
    country_name = country_data['country_name'].iloc[0]
    
    plt.figure(figsize=(14, 8))
    
    # Create a stacked area chart
    plt.stackplot(
        country_data['year'], 
        country_data['magnitude'],
        country_data['direction_change'],
        abs(country_data['acceleration']),
        labels=['Magnitude', 'Direction Change', '|Acceleration|'],
        alpha=0.7
    )
    
    # Add line for overall volatility
    plt.plot(
        country_data['year'], 
        country_data['volatility'], 
        color='black', 
        linewidth=2, 
        linestyle='--',
        label='Total Volatility'
    )
    
    plt.title(f'Volatility Component Breakdown for {country_code} ({country_name})')
    plt.xlabel('Year')
    plt.ylabel('Component Value')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, f'component_breakdown_{country_code}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_regional_comparison(df, output_dir):
    """Compare volatility patterns between different regions"""
    # Add basic region classification based on country code
    # This is a simplified approach - in reality you'd want a proper mapping
    region_map = {
        'NAM': ['USA', 'CAN', 'MEX'],
        'EUR': ['DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'NLD', 'CHE', 'SWE', 'POL'],
        'ASIA': ['CHN', 'JPN', 'IND', 'KOR', 'IDN', 'SGP', 'MYS', 'THA'],
        'MENA': ['SAU', 'ARE', 'QAT', 'IRN', 'ISR', 'TUR', 'EGY'],
        'AFR': ['ZAF', 'NGA', 'KEN', 'MAR', 'EGY'],
        'LATAM': ['BRA', 'ARG', 'CHL', 'COL', 'PER', 'MEX']
    }
    
    # Map countries to regions
    df['region'] = 'Other'
    for region, countries in region_map.items():
        df.loc[df['country_code'].isin(countries), 'region'] = region
    
    plt.figure(figsize=(14, 8))
    
    # Calculate average volatility by region and year
    region_data = df.groupby(['region', 'year'])['volatility'].mean().reset_index()
    
    # Plot line for each region
    sns.lineplot(
        data=region_data, 
        x='year', 
        y='volatility',
        hue='region',
        style='region',
        markers=True,
        dashes=False,
        linewidth=2
    )
    
    plt.title('Average Volatility by Region Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Volatility Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Region')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'regional_volatility_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_volatility_distributions(df, output_dir):
    """Plot the distribution of volatility metrics"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Distribution of volatility values
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df['volatility'], kde=True, ax=ax1)
    ax1.set_title('Distribution of Volatility Scores')
    ax1.set_xlabel('Volatility Score')
    ax1.set_ylabel('Frequency')
    
    # 2. Distribution of average volatility by country
    ax2 = fig.add_subplot(gs[0, 1])
    avg_by_country = df.groupby('country_code')['volatility'].mean()
    sns.histplot(avg_by_country, kde=True, ax=ax2)
    ax2.set_title('Distribution of Average Country Volatility')
    ax2.set_xlabel('Average Volatility Score')
    ax2.set_ylabel('Number of Countries')
    
    # 3. Scatterplot of magnitude vs direction change
    ax3 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(
        data=df,
        x='magnitude',
        y='direction_change',
        hue='volatility',
        palette='viridis',
        size='volatility',
        sizes=(20, 200),
        alpha=0.7,
        ax=ax3
    )
    ax3.set_title('Magnitude vs. Direction Change')
    ax3.set_xlabel('Magnitude')
    ax3.set_ylabel('Direction Change')
    
    # 4. Boxplot of volatility by decade
    ax4 = fig.add_subplot(gs[1, 1])
    df['decade'] = (df['year'] // 10) * 10
    sns.boxplot(data=df, x='decade', y='volatility', ax=ax4)
    ax4.set_title('Volatility by Decade')
    ax4.set_xlabel('Decade')
    ax4.set_ylabel('Volatility Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volatility_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_country_dashboard(df, output_dir, country_code):
    """Create a comprehensive dashboard for a single country"""
    country_data = df[df['country_code'] == country_code]
    if len(country_data) == 0:
        print(f"No data found for country code: {country_code}")
        return
    
    country_name = country_data['country_name'].iloc[0]
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Volatility timeline
    ax1 = fig.add_subplot(gs[0, 0])
    sns.lineplot(
        data=country_data,
        x='year',
        y='volatility',
        marker='o',
        color='#1f77b4',
        linewidth=2,
        ax=ax1
    )
    ax1.set_title(f'Volatility Timeline for {country_code} ({country_name})')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Volatility Score')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Component breakdown over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.stackplot(
        country_data['year'], 
        country_data['magnitude'],
        country_data['direction_change'],
        abs(country_data['acceleration']),
        labels=['Magnitude', 'Direction Change', '|Acceleration|'],
        alpha=0.7
    )
    ax2.plot(
        country_data['year'], 
        country_data['volatility'], 
        color='black', 
        linewidth=2, 
        linestyle='--',
        label='Total Volatility'
    )
    ax2.set_title('Component Breakdown')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Component Value')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Radar chart of components for latest year
    latest_year_data = country_data.loc[country_data['year'].idxmax()]
    categories = ['Volatility', 'Magnitude', 'Direction\nChange', 'Acceleration']
    values = [
        latest_year_data['volatility'],
        latest_year_data['magnitude'],
        latest_year_data['direction_change'],
        abs(latest_year_data['acceleration'])
    ]
    
    ax3 = fig.add_subplot(gs[1, 0], polar=True)
    
    # Close the polygon by appending the first value again
    values = np.concatenate((values, [values[0]]))
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax3.plot(angles, values, 'o-', linewidth=2)
    ax3.fill(angles, values, alpha=0.25)
    ax3.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax3.set_title(f'Component Profile ({latest_year_data["year"]})')
    
    # 4. Comparison with global average
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate global average by year
    global_avg = df.groupby('year')['volatility'].mean().reset_index()
    global_avg.columns = ['year', 'global_avg']
    
    # Merge with country data
    comparison_df = pd.merge(country_data[['year', 'volatility']], global_avg, on='year')
    
    # Calculate volatility relative to global average
    comparison_df['relative_volatility'] = comparison_df['volatility'] - comparison_df['global_avg']
    
    # Plot comparison
    sns.barplot(
        data=comparison_df,
        x='year',
        y='relative_volatility',
        ax=ax4,
        palette=['#d7191c' if x > 0 else '#2c7bb6' for x in comparison_df['relative_volatility']]
    )
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_title('Volatility Relative to Global Average')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Difference from Global Average')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    
    plt.suptitle(f'{country_code} ({country_name}) Volatility Dashboard', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    plt.savefig(os.path.join(output_dir, f'country_dashboard_{country_code}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Path to JSON data file
    json_path = "data/trajectory_clusters/country_yearly_volatility_weighted.json"
    
    # Output directory for visualizations
    output_dir = "visualizations/volatility_analysis"
    
    # Load data
    volatility_data = load_volatility_data(json_path)
    
    # Convert to DataFrame
    df = convert_to_dataframe(volatility_data)
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    plot_top_volatile_countries(df, output_dir, n=20)
    print("✓ Top volatile countries plot")
    
    plot_global_volatility_over_time(df, output_dir)
    print("✓ Global volatility trend plot")
    
    plot_volatility_components_heatmap(df, output_dir)
    print("✓ Volatility components correlation heatmap")
    
    # Custom selection of countries for comparison
    interesting_countries = ['USA', 'RUS', 'CHN', 'BRA', 'IND', 'ZAF', 'DEU', 'FRA', 'JPN', 'GBR']
    plot_country_comparison(df, output_dir, interesting_countries)
    print("✓ Country comparison plot")
    
    plot_volatility_heatmap(df, output_dir, n_countries=30)
    print("✓ Volatility heatmap")
    
    # Component breakdown for a few selected countries
    for country in ['USA', 'RUS', 'CHN', 'BRA']:
        if country in df['country_code'].unique():
            plot_component_breakdown(df, output_dir, country)
            print(f"✓ Component breakdown for {country}")
    
    plot_regional_comparison(df, output_dir)
    print("✓ Regional comparison plot")
    
    plot_volatility_distributions(df, output_dir)
    print("✓ Volatility distributions plots")
    
    # Create country dashboards for a few interesting countries
    for country in ['USA', 'RUS', 'CHN', 'BRA', 'DEU']:
        if country in df['country_code'].unique():
            plot_country_dashboard(df, output_dir, country)
            print(f"✓ Country dashboard for {country}")
    
    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()