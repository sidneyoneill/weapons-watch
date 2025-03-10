// src/components/ChartComponent.jsx
import React, { useRef, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register the required components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const ChartComponent = ({ timeSeriesData, selectedCountry, dataMode = 'total' }) => {
  const chartRef = useRef(null);

  // Clean up chart instance when component unmounts or data changes
  useEffect(() => {
    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, [timeSeriesData, selectedCountry, dataMode]);

  if (!timeSeriesData || timeSeriesData.length === 0) {
    return <div>No data available for {selectedCountry}</div>;
  }

  // Prepare chart data
  const years = timeSeriesData.map(entry => entry.Year || entry.year);
  const expenditures = timeSeriesData.map(entry => entry.Expenditure || entry.value);

  // Determine label and y-axis title based on data mode
  const labelText = dataMode === 'gdp' 
    ? `${selectedCountry} Military Expenditure (% of GDP)` 
    : `${selectedCountry} Military Expenditure (Constant USD)`;
  
  const yAxisText = dataMode === 'gdp' 
    ? 'Expenditure (% of GDP)' 
    : 'Expenditure (USD)';

  const data = {
    labels: years,
    datasets: [
      {
        label: labelText,
        data: expenditures,
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: yAxisText
        }
      },
      x: {
        title: {
          display: true,
          text: 'Year'
        }
      }
    }
  };

  return (
    <div className="chart-container">
      <h2>{selectedCountry} Military Expenditure Over Time</h2>
      <div style={{ height: '400px', width: '100%' }}>
        <Line 
          ref={chartRef}
          data={data} 
          options={options}
        />
      </div>
    </div>
  );
};

export default ChartComponent;
