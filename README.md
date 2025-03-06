# Arms Trade Dashboard

A comprehensive visualization tool for exploring global arms trade data. This project consists of a React frontend for interactive visualizations and a FastAPI backend for data processing and serving.

## Key Objectives & Questions

- Summary Stats
- Visualise military dependencies and alliances between countries. This would be Kmeans, etc
- Contextualise military expenditure and arms trading. Other datasets, News stories
- Prediction of trading/expenditure over the next few years?
- Conflict analysis

## Project Structure

arms-trade-dashboard/
├── frontend/ # React frontend application
├── backend/ # FastAPI backend server
├── data/ # Dataset files
├── notebooks/ # Jupyter notebooks for data analysis
└── venv/ # Python virtual environment

## Prerequisites

- Node.js (v14+) and npm for the frontend
- Python 3.8+ for the backend
- Git for version control

## Quick Start

Follow these steps to get the application running locally:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/arms-trade-dashboard.git
cd arms-trade-dashboard
```

### 2. Set up and start the backend

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt

# Start the backend server
cd backend
uvicorn app:app --reload
```

The backend API will be available at http://localhost:8000.

### 3. Set up and start the frontend

```bash
# In a new terminal, navigate to the frontend directory
cd frontend

# Install frontend dependencies
npm install

# Start the development server
npm run dev
```

The frontend application will be available at http://localhost:5173.

## Features

- Interactive world map showing arms trade data by country
- Time series charts of military expenditure
- Country-specific detailed analysis
- Responsive design for desktop and mobile viewing

## Documentation

- Backend API documentation is available at http://localhost:8000/docs when the backend is running
- See the [frontend README](./frontend/README.md) for more details on the frontend application

## License

[Your License Here]

## Acknowledgements

- SIPRI for providing the arms trade dataset
- [Any other acknowledgements]

## Available Scripts

- `npm start`: Runs the app in development mode
- `npm test`: Launches the test runner
- `npm run build`: Builds the app for production
