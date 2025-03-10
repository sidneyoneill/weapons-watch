import React from 'react';

/**
 * A toggle component that allows users to switch between "Total" and "% of GDP" data modes
 * 
 * @param {Object} props
 * @param {string} props.currentMode - Current data mode ('total' or 'gdp')
 * @param {function} props.onModeChange - Callback function when mode changes
 */
const DataModeToggle = ({ currentMode, onModeChange }) => {
  return (
    <div className="data-mode-toggle" style={styles.container}>
      <div style={styles.label}>Data Display Mode:</div>
      <div style={styles.buttonGroup}>
        <button
          style={{
            ...styles.button,
            ...(currentMode === 'total' ? styles.activeButton : {})
          }}
          onClick={() => onModeChange('total')}
        >
          Total Expenditure
        </button>
        <button
          style={{
            ...styles.button,
            ...(currentMode === 'gdp' ? styles.activeButton : {})
          }}
          onClick={() => onModeChange('gdp')}
        >
          % of GDP
        </button>
      </div>
    </div>
  );
};

// Inline styles for the component
const styles = {
  container: {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '20px',
    gap: '10px'
  },
  label: {
    fontWeight: 'bold',
    marginRight: '10px'
  },
  buttonGroup: {
    display: 'flex',
    borderRadius: '4px',
    overflow: 'hidden'
  },
  button: {
    padding: '8px 16px',
    border: '1px solid #ccc',
    background: '#f8f8f8',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    fontSize: '14px',
    outline: 'none'
  },
  activeButton: {
    background: '#4285f4',
    color: 'white',
    borderColor: '#4285f4'
  }
};

export default DataModeToggle; 