# Arms Trade Dashboard - Applied Data Science Project

[![Stars](https://img.shields.io/github/stars/sidneyoneill/arms-trade-dashboard?style=social)](https://github.com/sidneyoneill/arms-trade-dashboard)
[![Forks](https://img.shields.io/github/forks/sidneyoneill/arms-trade-dashboard?style=social)](https://github.com/sidneyoneill/arms-trade-dashboard)

## Project Title & Description

This project aims to provide an interactive dashboard for visualizing and analyzing global arms trade data. It leverages various data sources, including SIPRI, ACLED, and GDELT, to create a comprehensive picture of arms imports, exports, military expenditure, and their correlation with conflict events. The dashboard utilizes data science and machine learning techniques to uncover hidden patterns and trends, making the complex arms trade landscape more accessible and understandable.

## Key Features & Benefits

- **Interactive Visualizations:** Dynamic maps, charts, and chord diagrams for exploring arms trade relationships.
- **Data Integration:** Combines data from multiple reputable sources for a holistic view.
- **Time-Series Analysis:** Analyze trends in arms trade and military expenditure over time.
- **Geospatial Analysis:** Visualize arms trade flows and military spending on a global scale.
- **Clustering Analysis:** Uncover groups of countries with similar arms trade trajectories.
- **Dimensionality Reduction:** Simplify complex datasets for better visualization and analysis.
- **Data Mode Toggle:** Allows users to switch between different data representations.
- **Year Slider:** Enables users to explore data for specific years.
- **Front-end built with React, making it interactive and user-friendly.**
- **Back-end built with Python, providing data processing and machine learning capabilities.**

## Technologies

### Languages

- Python
- JavaScript
- TypeScript

### Tools & Technologies

- Node.js
- React
- Vite
- D3.js
- Flask
- Pandas
- Scikit-learn
- GeoPandas
- NumPy

## Prerequisites & Dependencies

Before you begin, ensure you have the following installed:

- **Python (>=3.7):** Required for backend processing and analysis.
- **Node.js (>=16):** Required for frontend development.
- **pip:** Python package installer. Comes standard with most Python installations.
- **npm:** Node Package Manager. Comes standard with Node.js installations.
- **Conda (Optional but Recommended):** For managing Python environments.

## Installation & Setup Instructions

Follow these steps to get the project up and running:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/sidneyoneill/arms-trade-dashboard.git
    cd arms-trade-dashboard
    ```

2.  **Set up the Python environment (Recommended: Use Conda):**

    - **Using Conda:**

      ```bash
      conda env create -f environment.yml
      conda activate arms-trade-dashboard-env
      ```

      If the env name in yml file is not properly set you may have to change the `conda activate` command.

    - **Using pip (Alternative):**

      ```bash
      python -m venv venv
      source venv/bin/activate  # On Linux/macOS
      # venv\Scripts\activate  # On Windows

      pip install -r requirements.txt
      ```

      (You may need to generate requirements.txt first. See `environment.yml` for a guide.)

3.  **Set up the Frontend (Node.js):**

    ```bash
    cd frontend
    npm install
    ```

4.  **Run the Backend:**

    ```bash
    cd ../backend
    python app.py
    ```

    This will start the Flask server, typically on `http://127.0.0.1:5000`.

5.  **Run the Frontend:**

    ```bash
    cd ../frontend
    npm run dev
    ```

    This will start the Vite development server, typically on `http://localhost:5173`. The exact port might vary. Access this URL in your browser to view the dashboard.

## Project Structure

```
├── .DS_Store
├── .cursorignore
├── .gitignore
├── Arms_Trade_Data.md
├── README.md
├── environment.yml
├── gantt_chart.png
├── network_diagram.png
├── package-lock.json
├── package.json
├── analysis
│   ├── ACLED_Exploration.ipynb
│   ├── GDELT_Exloration.ipynb
│   ├── SIPRI_Milex_Exploration.ipynb
│   ├── SIPRI_Trade_Chord_Diagrams.ipynb
│   ├── SIPRI_Trade_Exploration.ipynb
│   ├── clustering.ipynb
│   └── project_management
│       ├── gantt_chart.png
│       ├── gantt_chart.py
│       ├── network_diagram.png
│       └── network_diagram.py
├── backend
│   ├── app.py
│   ├── csv_to_json.py
│   ├── prepare_arms_trade_matrix.py
│   ├── preprocess.py
│   ├── preprocess_master_dataset.py
│   ├── preprocess_trade_data.py
│   └── ml
│       ├── clustering.py
│       ├── dim_reduct.ipynb
│       ├── dimensionality_reduction.py
│       └── vector_clustering.py
├── data
│   ├── .DS_Store
│   ├── all_data_merged.json
│   ├── all_data_merged_cleaned.json
│   ├── sipri_milex_data.csv
│   ├── sipri_milex_data_merged.geojson
│   ├── sipri_milex_data_nested.json
│   ├── sipri_milex_data_tidy.csv
│   ├── sipri_milex_gdp_data.csv
│   ├── sipri_milex_gdp_data_merged.geojson
│   ├── sipri_milex_gdp_data_tidy.csv
│   ├── sipri_trade_data.csv
│   ├── sipri_trade_data_tidy.csv
│   ├── world_countries.geojson
│   ├── arms_trade_matrices
│   │   ├── arms_trade_matrices_all_years.json
│   │   ├── arms_trade_matrix_2015.json
│   │   ├── arms_trade_matrix_2016.json
│   │   ├── arms_trade_matrix_2017.json
│   │   ├── arms_trade_matrix_2018.json
│   │   ├── arms_trade_matrix_2019.json
│   │   ├── arms_trade_matrix_2020.json
│   │   └── arms_trade_matrix_2021.json
│   ├── clustering_results
│   │   ├── all_countries_tsne.csv
│   │   ├── all_countries_tsne.json
│   │   ├── tsne_clusters_dim1_2.png
│   │   ├── tsne_clusters_dim1_3.png
│   │   └── tsne_clusters_dim2_3.png
│   ├── dimensionality_reduction_g20
│   │   ├── dim_reduction_metadata.json
│   │   ├── dim_reduction_pca_3d_plot.png
│   │   ├── dim_reduction_pca_loadings_2d.png
│   │   ├── dim_reduction_pca_loadings_3d.png
│   │   ├── dim_reduction_pca_plot.png
│   │   ├── dim_reduction_pca_results.json
│   │   ├── dim_reduction_tsne_3d_plot.png
│   │   ├── dim_reduction_tsne_plot.png
│   │   └── dim_reduction_tsne_results.json
│   ├── trajectory_clusters
│   │   ├── cluster_feature_distributions.png
│   │   ├── cluster_radar_profiles.png
│   │   ├── trajectory_clusters.csv
│   │   ├── trajectory_clusters.json
│   │   ├── trajectory_clusters_dim1_2_cluster_0.png
│   │   ├── trajectory_clusters_dim1_2_cluster_1.png
│   │   ├── trajectory_clusters_dim1_2_cluster_2.png
│   │   ├── trajectory_clusters_dim1_2_cluster_3.png
│   │   ├── trajectory_clusters_dim1_2_cluster_4.png
│   │   ├── trajectory_clusters_dim1_3_cluster_0.png
│   │   ├── trajectory_clusters_dim1_3_cluster_1.png
│   │   ├── trajectory_clusters_dim1_3_cluster_2.png
│   │   ├── trajectory_clusters_dim1_3_cluster_3.png
│   │   ├── trajectory_clusters_dim1_3_cluster_4.png
│   │   ├── trajectory_clusters_dim2_3_cluster_0.png
│   │   ├── trajectory_clusters_dim2_3_cluster_1.png
│   │   ├── trajectory_clusters_dim2_3_cluster_2.png
│   │   ├── trajectory_clusters_dim2_3_cluster_3.png
│   │   └── trajectory_clusters_dim2_3_cluster_4.png
│   └── world_bank
│       ├── WDICSV_modified.csv
│       ├── exploration.ipynb
│       ├── json_creator.py
│       ├── world_bank_data_normalized.json
│       └── feature_selection_plots
│           ├── Armed forces personnel, total.png
│           ├── Arms exports (SIPRI trend indicator values).png
│           ├── Arms imports (SIPRI trend indicator values).png
│           ├── BM.KLT.DINV.WD.GD.ZS.png
│           ├── Foreign direct investment, net outflows (% of GDP).png
│           ├── Fossil fuel energy consumption (% of total).png
│           ├── GDP growth (annual %).png
│           ├── IT.NET.USER.ZS.png
│           ├── Individuals using the Internet (% of population).png
│           ├── Internally displaced persons, total displaced by conflict and violence (number of people).png
│           ├── International migrant stock (% of population).png
│           ├── MS.MIL.MPRT.KD.png
│           ├── MS.MIL.TOTL.P1.png
│           ├── MS.MIL.XPRT.KD.png
│           ├── NY.GDP.MKTP.KD.ZG.png
│           ├── Ores and metals exports (% of merchandise exports).png
│           ├── SM.POP.TOTL.ZS.png
│           ├── TX.VAL.MMTL.ZS.UN.png
│           └── VC.IDP.TOCV.png
├── frontend
│   ├── .gitignore
│   ├── README.md
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── vite.config.js
│   └── src
│       ├── App.jsx
│       ├── index.css
│       ├── main.jsx
│       └── components
│           ├── ArmsTradeDashboard.jsx
│           ├── ChartComponent.jsx
│           ├── ChordDiagram.jsx
│           ├── CountryExpidentureComponent.jsx
│           ├── DataModeToggle.jsx
│           ├── ExpidentureMapComponent.jsx
│           ├── MapComponent.jsx
│           └── YearSlider.jsx
└── node_modules
    ├── ... (truncated for brevity)
```

## Usage Examples & API Documentation

### Backend API

The backend provides endpoints for fetching data and performing analysis.

- `/data`: Returns arms trade data in JSON format. Supports filtering by year and country.
- `/clustering`: Returns clustering results based on specified parameters.

Refer to the `backend/app.py` file for a complete list of endpoints and their parameters.

### Frontend Components

The frontend utilizes React components for creating the dashboard.

- `MapComponent`: Displays a world map with arms trade data visualized.
- `ChartComponent`: Renders charts and graphs for analyzing trends.
- `ChordDiagram`: Shows the relationships between arms importers and exporters.
- `YearSlider`: A slider component to select the year for data visualization.

Refer to the `frontend/src/components` directory for component documentation and usage examples.

## Configuration Options

### Backend

- **Port:** The Flask server runs on port `5000` by default. You can change this in `backend/app.py`.
- **Data Sources:** The application reads data from the `data` directory. Modify the paths in the Python scripts to use different data files.

### Frontend

- **API Endpoint:** The frontend connects to the backend API at `http://127.0.0.1:5000` by default. You can change this in `frontend/src/App.jsx`.
- **Map Styling:** Modify the styling of the map in `frontend/src/components/MapComponent.jsx`.

## Contributing Guidelines

We welcome contributions to this project! Please follow these guidelines:

1.  **Fork the repository.**
2.  **Create a new branch for your feature or bug fix.**
3.  **Make your changes and commit them with descriptive commit messages.**
4.  **Submit a pull request to the main branch.**

Please ensure your code adheres to the project's coding style and includes appropriate tests.

## License Information

License not specified. All rights reserved to owner.

## Acknowledgments

- Stockholm International Peace Research Institute (SIPRI) for providing valuable arms trade data.
- Armed Conflict Location & Event Data Project (ACLED) for providing conflict event data.
- Global Database of Events, Language, and Tone (GDELT) for providing event data.
- D3.js library for data visualization.
- React library for front-end development.

```

```
