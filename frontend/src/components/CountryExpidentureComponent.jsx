import React, { useState } from "react";
import MapComponent from "./MapComponent";
import ChartComponent from "./ChartComponent";

const CountryExpenditureComponent = () => {
  const [selectedCountry, setSelectedCountry] = useState(null);
  const [timeSeriesData, setTimeSeriesData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleCountrySelect = (country) => {
    setSelectedCountry(country);
    setError(null);
    setLoading(true);

    fetch(`http://localhost:8000/expenditure/${encodeURIComponent(country)}`)
      .then((response) => response.json())
      .then((data) => {
        if (data && data.time_series && data.time_series.length > 0) {
          setTimeSeriesData(data.time_series);
        } else {
          setError("No data available for " + country);
        }
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        setError("Failed to fetch data for " + country);
        setLoading(false);
      });
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Select a Country to View Expenditure Trends</h2>
      <MapComponent onCountrySelect={handleCountrySelect} />

      {loading && <div>Loading data...</div>}
      {error && <div style={{ color: "red" }}>{error}</div>}

      {selectedCountry && !loading && timeSeriesData.length > 0 && (
        <div style={{ marginTop: "30px" }}>
          <ChartComponent
            timeSeriesData={timeSeriesData}
            selectedCountry={selectedCountry}
          />
        </div>
      )}
    </div>
  );
};

export default CountryExpenditureComponent;
