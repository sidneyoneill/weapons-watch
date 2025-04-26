import React from 'react';

function YearSlider({ selectedYear, onYearChange, minYear, maxYear }) {
  return (
    <div className="year-slider">
      <label htmlFor="year-select">Select Year: </label>
      <input
        type="range"
        id="year-select"
        min={minYear}
        max={maxYear}
        value={selectedYear}
        onChange={(e) => onYearChange(parseInt(e.target.value))}
      />
      <span>{selectedYear}</span>
    </div>
  );
}

export default YearSlider;
