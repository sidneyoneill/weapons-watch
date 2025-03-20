import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create a new graph
G = nx.Graph()

# Define nodes for objectives and activities
objectives = {
    "O1": "Visualize Arms Trading Effectively",
    "O2": "Collect and Analyze Development Indicators",
    "O3": "Explore Impact of Development on Arms Trading",
    "INT": "Integration: Draw Conclusions & Publish Findings"
}

activities = {
    # Objective 1 activities
    "A1.1": "Collect SIPRI Arms Trade Data",
    "A1.2": "Build Multi-view Visualizations",
    
    # Objective 2 activities
    "A2.1": "Collect World Bank Data",
    "A2.2": "Feature Selection and Cleaning",
    "A2.3": "Cluster and Visualize Development Trajectories",
    
    # Objective 3 activities
    "A3.1": "Apply Clustering Techniques",
    "A3.2": "Perform Dimensionality Reduction",
    "A3.3": "Develop Predictive Models"
}

# Add nodes with their attributes
# Objectives (larger nodes)
G.add_node("O1", pos=(5, 10), size=1500, color="#3498db", type="objective")
G.add_node("O2", pos=(2, 6), size=1500, color="#e74c3c", type="objective")
G.add_node("O3", pos=(8, 6), size=1500, color="#2ecc71", type="objective")
G.add_node("INT", pos=(5, 1), size=1200, color="#9b59b6", type="integration")

# Activities (smaller nodes)
# Objective 1 activities
G.add_node("A1.1", pos=(4, 8), size=800, color="#3498db", type="activity")
G.add_node("A1.2", pos=(6, 8), size=800, color="#3498db", type="activity")

# Objective 2 activities
G.add_node("A2.1", pos=(1, 4), size=800, color="#e74c3c", type="activity")
G.add_node("A2.2", pos=(3, 4), size=800, color="#e74c3c", type="activity")
G.add_node("A2.3", pos=(2, 2), size=800, color="#e74c3c", type="activity")

# Objective 3 activities
G.add_node("A3.1", pos=(7, 4), size=800, color="#2ecc71", type="activity")
G.add_node("A3.2", pos=(9, 4), size=800, color="#2ecc71", type="activity")
G.add_node("A3.3", pos=(8, 2), size=800, color="#2ecc71", type="activity")

# Add edges between nodes
# Objective 1 connections
G.add_edge("O1", "A1.1", weight=3, color="#3498db", style="solid")
G.add_edge("O1", "A1.2", weight=3, color="#3498db", style="solid")
G.add_edge("A1.1", "A1.2", weight=2, color="#3498db", style="dashed")

# Objective 2 connections
G.add_edge("O2", "A2.1", weight=3, color="#e74c3c", style="solid")
G.add_edge("O2", "A2.2", weight=3, color="#e74c3c", style="solid")
G.add_edge("O2", "A2.3", weight=3, color="#e74c3c", style="solid")
G.add_edge("A2.1", "A2.2", weight=2, color="#e74c3c", style="dashed")
G.add_edge("A2.2", "A2.3", weight=2, color="#e74c3c", style="dashed")

# Objective 3 connections
G.add_edge("O3", "A3.1", weight=3, color="#2ecc71", style="solid")
G.add_edge("O3", "A3.2", weight=3, color="#2ecc71", style="solid")
G.add_edge("O3", "A3.3", weight=3, color="#2ecc71", style="solid")
G.add_edge("A3.1", "A3.2", weight=2, color="#2ecc71", style="dashed")
G.add_edge("A3.1", "A3.3", weight=2, color="#2ecc71", style="dashed")
G.add_edge("A3.2", "A3.3", weight=2, color="#2ecc71", style="dashed")

# Integration connections
G.add_edge("A2.3", "INT", weight=3, color="#9b59b6", style="solid")
G.add_edge("A3.3", "INT", weight=3, color="#9b59b6", style="solid")

# Cross-objective connections
G.add_edge("A1.1", "O2", weight=2, color="#9b59b6", style="dashed")
G.add_edge("A1.2", "O3", weight=2, color="#9b59b6", style="dashed")
G.add_edge("A2.2", "A3.1", weight=2, color="#9b59b6", style="dashed")

# Extract node positions, colors, and sizes
pos = nx.get_node_attributes(G, 'pos')
node_colors = [G.nodes[n]['color'] for n in G.nodes()]
node_sizes = [G.nodes[n]['size'] for n in G.nodes()]

# Extract edge colors and styles
edge_colors = [G[u][v]['color'] for u, v in G.edges()]
edge_styles = [G[u][v]['style'] for u, v in G.edges()]
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

# Create figure and axis
plt.figure(figsize=(16, 12))
plt.title("Arms Trading and Development Indicators Project: Network Diagram", fontsize=20, pad=20)

# Draw the network
solid_edges = [(u, v) for u, v in G.edges() if G[u][v]['style'] == 'solid']
dashed_edges = [(u, v) for u, v in G.edges() if G[u][v]['style'] == 'dashed']

solid_colors = [G[u][v]['color'] for u, v in solid_edges]
dashed_colors = [G[u][v]['color'] for u, v in dashed_edges]

solid_weights = [G[u][v]['weight'] for u, v in solid_edges]
dashed_weights = [G[u][v]['weight'] for u, v in dashed_edges]

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

# Draw edges with different styles
nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=solid_weights, 
                       edge_color=solid_colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, width=dashed_weights, 
                       edge_color=dashed_colors, style='dashed', alpha=0.8)

# Add labels with custom positions
label_pos = {k: (v[0], v[1] - 0.2) for k, v in pos.items()}
nx.draw_networkx_labels(G, label_pos, 
                        labels={n: n for n in G.nodes() if G.nodes[n]['type'] != 'integration'},
                        font_size=10, font_weight='bold')

# Add a more detailed description for the integration node
nx.draw_networkx_labels(G, {k: (v[0], v[1] - 0.2) for k, v in pos.items() if k == 'INT'}, 
                        labels={'INT': 'INT'}, font_size=10, font_weight='bold')

# Add descriptions below node labels
for node, (x, y) in pos.items():
    if node in objectives:
        plt.text(x, y-0.4, objectives[node], 
                 horizontalalignment='center', fontsize=8, wrap=True)
    elif node in activities:
        plt.text(x, y-0.4, activities[node], 
                 horizontalalignment='center', fontsize=8, wrap=True)

# Create legend
legend_elements = [
    mpatches.Patch(color="#3498db", alpha=0.8, label="Objective 1: Visualize Arms Trading"),
    mpatches.Patch(color="#e74c3c", alpha=0.8, label="Objective 2: Development Indicators"),
    mpatches.Patch(color="#2ecc71", alpha=0.8, label="Objective 3: Impact Analysis"),
    mpatches.Patch(color="#9b59b6", alpha=0.8, label="Integration Activities")
]
plt.legend(handles=legend_elements, loc="upper left", fontsize=12)

# Remove axis
plt.axis('off')

# Save figure with high resolution
plt.savefig('network_diagram.png', dpi=300, bbox_inches='tight')
plt.show()