import React from 'react';

// Theme colors to match App.jsx
const theme = {
  background: '#000000',
  secondaryBg: '#111111',
  accent: '#ea580c', // Orange accent matching the globe
  text: {
    primary: '#ffffff',
    secondary: '#cccccc',
  },
  borders: '#333333',
};

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
      <div style={styles.label}>Data View:</div>
      <div style={styles.buttonGroup}>
        <button
          style={{
            ...styles.button,
            ...(currentMode === 'total' ? styles.activeButton : {})
          }}
          onClick={() => onModeChange('total')}
        >
          Total
        </button>
        <button
          style={{
            ...styles.button,
            ...(currentMode === 'gdp' ? styles.activeButton : {})
          }}
          onClick={() => onModeChange('gdp')}
        >
          % GDP
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
    gap: '10px'
  },
  label: {
    fontWeight: '500',
    color: theme.text.secondary,
    fontSize: '14px'
  },
  buttonGroup: {
    display: 'flex',
    borderRadius: '4px',
    overflow: 'hidden',
    border: `1px solid ${theme.borders}`,
  },
  button: {
    padding: '6px 12px',
    border: 'none',
    borderRight: `1px solid ${theme.borders}`,
    background: 'rgba(20, 20, 20, 0.6)',
    color: theme.text.secondary,
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    fontSize: '13px',
    fontWeight: '500',
    outline: 'none',
    '&:last-child': {
      borderRight: 'none'
    }
  },
  activeButton: {
    background: theme.accent,
    color: 'white',
    boxShadow: '0 0 10px rgba(234, 88, 12, 0.5)'
  }
};

export default DataModeToggle;