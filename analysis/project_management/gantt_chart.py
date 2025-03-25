import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define project data with simplified key activities
data = {
    'Task': [
        # Objective 1: Visualize Arms Trading
        'Arms Data Collection & Processing',
        'Trading Visualization Development',
        
        # Objective 2: Development Indicators
        'Development Indicators Collection',
        'Indicator Analysis & Clustering',
        
        # Objective 3: Impact Analysis
        'Analysis Framework Development',
        'Correlation & Trajectory Analysis',
        'Predictive Modeling & Validation',
        
        # Integration
        'Integration & Documentation',
        'Final Report & Presentation'
    ],
    
    'Start': [
        # Objective 1 (completed)
        '2025-02-27', '2025-03-06',
        
        # Objective 2 (completed)
        '2025-03-01', '2025-03-10',
        
        # Objective 3 (just started)
        '2025-03-18', '2025-03-25', '2025-04-05',
        
        # Integration
        '2025-04-08', '2025-04-13'
    ],
    
    'Duration': [
        # Objective 1
        7, 8,
        
        # Objective 2
        9, 8,
        
        # Objective 3
        7, 11, 8,
        
        # Integration
        5, 4
    ],
    
    'Category': [
        # Objective 1
        'Objective 1', 'Objective 1',
        
        # Objective 2
        'Objective 2', 'Objective 2',
        
        # Objective 3
        'Objective 3', 'Objective 3', 'Objective 3',
        
        # Integration
        'Integration', 'Integration'
    ],
    
    'Completion': [
        # Completed work
        100, 100,  # Objective 1
        100, 100,  # Objective 2
        
        # Current & future work
        10, 0, 0,  # Objective 3 (just started)
        0, 0       # Integration
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert string dates to datetime
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = df['Start'] + pd.to_timedelta(df['Duration'], unit='D')

# Define colors for categories
colors = {
    'Objective 1': '#3498db',
    'Objective 2': '#e74c3c',
    'Objective 3': '#2ecc71',
    'Integration': '#9b59b6'
}

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(left=0.25)

# Set axis limits
ylim = (0, len(df['Task']))
ax.set_ylim(ylim)
ax.set_xlim([pd.Timestamp('2025-02-25'), pd.Timestamp('2025-04-20')])  # Set visible range

# Create Y-axis with task names
ax.set_yticks(range(len(df['Task'])))
ax.set_yticklabels(df['Task'])
ax.tick_params(axis='y', labelsize=11)

# Format x-axis dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
plt.xticks(rotation=45, ha='right')

# Set grid
ax.grid(True, axis='x', alpha=0.3)

# Draw Gantt bars
for i, task in enumerate(df['Task']):
    start_date = df['Start'][i]
    end_date = df['End'][i]
    duration = (end_date - start_date).days
    category = df['Category'][i]
    completion = df['Completion'][i]
    
    # Draw full task bar (background)
    ax.barh(
        i, 
        duration, 
        left=start_date, 
        height=0.5, 
        color=colors[category], 
        alpha=0.4,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Draw completion portion
    if completion > 0:
        completed_width = duration * (completion / 100)
        ax.barh(
            i,
            completed_width,
            left=start_date,
            height=0.5,
            color=colors[category],
            alpha=0.9,
            edgecolor='black',
            linewidth=0.5
        )
    
    # Add task duration and completion text
    plt.text(
        start_date + timedelta(days=duration/2), 
        i, 
        f"{duration}d ({completion}%)",
        ha='center',
        va='center',
        color='black',
        fontweight='bold'
    )

# Add category labels
category_positions = {}
for category in ['Objective 1', 'Objective 2', 'Objective 3', 'Integration']:
    category_tasks = df[df['Category'] == category]
    if len(category_tasks) > 0:
        start_pos = df[df['Category'] == category].index[0]
        end_pos = df[df['Category'] == category].index[-1] + 1
        
        # Store the midpoint of each category section
        category_positions[category] = (start_pos + end_pos) / 2
        
        # Add separator line except for the first category
        if category != 'Objective 1':
            ax.axhline(y=start_pos - 0.5, color='black', linestyle='-', alpha=0.3)

# Add a legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=colors['Objective 1'], lw=10, alpha=0.8, label='Objective 1: Arms Trading Visualization'),
    Line2D([0], [0], color=colors['Objective 2'], lw=10, alpha=0.8, label='Objective 2: Development Indicators'),
    Line2D([0], [0], color=colors['Objective 3'], lw=10, alpha=0.8, label='Objective 3: Impact Analysis'),
    Line2D([0], [0], color=colors['Integration'], lw=10, alpha=0.8, label='Integration & Final Deliverables')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

# Add today's date marker
today = pd.Timestamp('2025-03-20')
ax.axvline(x=today, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(today, len(df) + 0.3, 'Today', ha='center', va='bottom', color='red', fontweight='bold')

# Add project end marker
project_end = pd.Timestamp('2025-04-17')
ax.axvline(x=project_end, color='blue', linestyle='--', linewidth=2, alpha=0.7)
ax.text(project_end, len(df) + 0.3, 'Deadline', ha='center', va='bottom', color='blue', fontweight='bold')

# Add key dependencies
dependencies = [
    ('Trading Visualization Development', 'Analysis Framework Development'),
    ('Indicator Analysis & Clustering', 'Correlation & Trajectory Analysis'),
    ('Predictive Modeling & Validation', 'Integration & Documentation')
]

# Add dependency arrows
for source, target in dependencies:
    source_idx = df[df['Task'] == source].index[0]
    source_end = df[df['Task'] == source]['End'].iloc[0]
    
    target_idx = df[df['Task'] == target].index[0]
    target_start = df[df['Task'] == target]['Start'].iloc[0]
    
    ax.annotate(
        '',
        xy=(target_start, target_idx),
        xytext=(source_end, source_idx),
        arrowprops=dict(
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.2',
            color='gray',
            alpha=0.8,
            linewidth=1.5
        )
    )

# Adjust layout and save
plt.tight_layout()
plt.savefig('gantt_chart.png', dpi=300, bbox_inches='tight')
plt.show()