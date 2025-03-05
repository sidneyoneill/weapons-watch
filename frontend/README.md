# Arms Trade Dashboard Frontend

The frontend application for the Arms Trade Dashboard, built with React and Vite.

## Technology Stack

- **React**: UI library
- **Vite**: Build tool and development server
- **Chart.js & react-chartjs-2**: For data visualization
- **Leaflet & react-leaflet**: For interactive maps
- **Axios**: For API requests

## Getting Started

### Prerequisites

- Node.js (v14+)
- npm or yarn

### Installation

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install
```

### Development

```bash
# Start the development server
npm run dev
```

This will start the development server at http://localhost:5173.

### Building for Production

```bash
# Create a production build
npm run build

# Preview the production build locally
npm run preview
```

The production build will be in the `dist` directory.

## Project Structure

```
frontend/
├── public/           # Static assets
├── src/              # Source code
│   ├── components/   # React components
│   │   ├── MapComponent.jsx    # Interactive map
│   │   └── ChartComponent.jsx  # Data charts
│   ├── App.jsx       # Main application component
│   └── main.jsx      # Application entry point
├── index.html        # HTML template
├── vite.config.js    # Vite configuration
└── package.json      # Project dependencies and scripts
```

## Component Overview

### MapComponent

The `MapComponent` displays an interactive world map using react-leaflet. Countries can be clicked to view their military expenditure data.

### ChartComponent

The `ChartComponent` displays time series data for military expenditure using Chart.js.

## API Integration

The frontend communicates with the backend API at `http://localhost:8000` to fetch:

- GeoJSON data for the world map
- Time series data for military expenditure by country

## Customization

### Styling

The application uses CSS for styling. Main styles are in `src/index.css`.

### Map Configuration

Map settings can be adjusted in `src/components/MapComponent.jsx`.

### Chart Configuration

Chart settings can be adjusted in `src/components/ChartComponent.jsx`.

## Troubleshooting

### Common Issues

1. **Map doesn't render**: Ensure the backend is running and CORS is properly configured.
2. **Chart errors**: Check that Chart.js components are properly registered.
3. **No data displayed**: Verify the backend API is returning the expected data format.

### Browser Console

Check the browser console (F12) for error messages that might help diagnose issues.