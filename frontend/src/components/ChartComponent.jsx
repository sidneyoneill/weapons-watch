// src/components/ChartComponent.jsx
import React, { useRef, useEffect, useState } from 'react';
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

// Define theme colors to match the app
const theme = {
  background: 'rgba(17, 17, 17, 0.5)',
  accent: '#ea580c', // Orange accent color
  text: {
    primary: '#ffffff',
    secondary: '#cccccc',
  },
  grid: '#333333',
};

const ChartComponent = ({ timeSeriesData, selectedCountry, dataMode = 'total' }) => {
  const chartRef = useRef(null);
  const [chartInstance, setChartInstance] = useState(null);
  
  // Check if we have valid data to display
  const hasData = Array.isArray(timeSeriesData) && timeSeriesData.length > 0;
  
  // Clean up chart instance when component unmounts or data changes
  useEffect(() => {
    // Cleanup function to destroy chart when unmounting or when data changes
    return () => {
      if (chartInstance) {
        chartInstance.destroy();
      }
    };
  }, [chartInstance, timeSeriesData, selectedCountry, dataMode]);

  // If no data is available, show a message
  if (!hasData) {
    return <div style={{ color: theme.text.secondary, textAlign: 'center', padding: '20px' }}>
      No data available for {selectedCountry}
    </div>;
  }

  // Prepare chart data
  const years = timeSeriesData.map(entry => entry.Year || entry.year);
  const expenditures = timeSeriesData.map(entry => entry.Expenditure || entry.military_expenditure || entry.value);
  
  // Determine label and y-axis title based on data mode
  const labelText = dataMode === 'gdp' 
    ? `Military Expenditure (% of GDP)` 
    : `Military Expenditure (USD millions)`;
  
  const yAxisText = dataMode === 'gdp' 
    ? '% of GDP' 
    : 'USD (millions)';

  const data = {
    labels: years,
    datasets: [
      {
        label: labelText,
        data: expenditures,
        fill: true,
        borderColor: theme.accent,
        backgroundColor: `${theme.accent}33`, // Adding alpha for transparency
        tension: 0.2,
        borderWidth: 2,
        pointBackgroundColor: theme.accent,
        pointBorderColor: '#fff',
        pointRadius: 4,
        pointHoverRadius: 6,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: dataMode === 'gdp' ? true : false, // Start at zero for percentage, not for absolute values
        title: {
          display: true,
          text: yAxisText,
          color: theme.text.secondary,
          font: {
            size: 12
          }
        },
        grid: {
          color: theme.grid,
          borderColor: theme.grid,
        },
        ticks: {
          color: theme.text.secondary,
          callback: function(value) {
            if (dataMode === 'gdp') {
              return value + '%';
            } else {
              return value >= 1000 ? `$${value/1000}B` : `$${value}M`;
            }
          }
        }
      },
      x: {
        title: {
          display: true,
          text: 'Year',
          color: theme.text.secondary,
          font: {
            size: 12
          }
        },
        grid: {
          color: theme.grid,
          borderColor: theme.grid,
        },
        ticks: {
          color: theme.text.secondary,
          maxRotation: 45,
          minRotation: 45
        }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          color: theme.text.primary,
          font: {
            size: 12
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(17, 17, 17, 0.9)',
        titleColor: theme.accent,
        bodyColor: theme.text.primary,
        titleFont: {
          size: 14,
          weight: 'bold'
        },
        bodyFont: {
          size: 13
        },
        borderColor: theme.accent,
        borderWidth: 1,
        padding: 10,
        displayColors: false,
        callbacks: {
          title: function(tooltipItems) {
            return `Year: ${tooltipItems[0].label}`;
          },
          label: function(context) {
            let value = context.parsed.y;
            if (dataMode === 'gdp') {
              return `Military Expenditure: ${value.toFixed(2)}% of GDP`;
            } else {
              return `Military Expenditure: $${value.toLocaleString()} million`;
            }
          }
        }
      }
    },
    animation: {
      duration: 1000
    }
  };

  return (
    <div className="chart-container">
      <div style={{ 
        height: '300px', 
        width: '100%',
        position: 'relative',
        padding: '10px',
        backgroundColor: theme.background,
        borderRadius: '4px',
        border: `1px solid ${theme.grid}`
      }}>
        {hasData && (
          <Line 
            data={data} 
            options={options}
            ref={instance => {
              chartRef.current = instance?.canvas;
              setChartInstance(instance?.chartInstance);
            }}
          />
        )}
      </div>
    </div>
  );
};

export default ChartComponent;
