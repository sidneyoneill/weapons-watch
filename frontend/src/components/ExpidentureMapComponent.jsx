import React, { useEffect, useState } from "react";
import "leaflet/dist/leaflet.css";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  LayerGroup,
} from "react-leaflet";
import axios from "axios";
import L from "leaflet";
import ChartComponent from "./ChartComponent";

const ExpenditureMapComponent = ({ dataMode = 'total' }) => {
  const [geoData, setGeoData] = useState(null);
  const [expenditureData, setExpenditureData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedYear, setSelectedYear] = useState(2000); // Default year
  const [maxExpenditure, setMaxExpenditure] = useState(1); // For scaling circles
  const [selectedCountry, setSelectedCountry] = useState(null); // Track clicked country
  const [countryHistoryData, setCountryHistoryData] = useState([]); // Store historical data

  // Fetch geo data when component mounts or dataMode changes
  useEffect(() => {
    setLoading(true);
    axios
      .get(`http://localhost:8000/geo_data?mode=${dataMode}`)
      .then((response) => {
        try {
          const parsedData = JSON.parse(response.data.data);
          setGeoData(parsedData);
          fetchAllExpenditureData();
        } catch (e) {
          console.error("Error parsing geo data:", e);
          setError("Failed to parse geo data");
          setLoading(false);
        }
      })
      .catch((error) => {
        console.error("Error fetching geo data:", error);
        setError("Failed to fetch geo data");
        setLoading(false);
      });
  }, [dataMode]); // Re-fetch when dataMode changes

  // Fetch expenditure data for all countries
  const fetchAllExpenditureData = async () => {
    try {
      const response = await axios.get(
        `http://localhost:8000/all_expenditures?mode=${dataMode}`
      );
      const data = response.data.time_series;
      console.log("Received expenditure data:", data);

      // Process data to organize by year and find max value
      const organizedData = {};
      let maxValue = 0;

      data.forEach((item) => {
        const { Country, Year, Expenditure } = item;

        if (!organizedData[Year]) {
          organizedData[Year] = {};
        }

        organizedData[Year][Country] = Expenditure;

        if (Expenditure > maxValue) {
          maxValue = Expenditure;
        }
      });

      setExpenditureData(organizedData);
      setMaxExpenditure(maxValue);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching expenditure data:", error);
      setLoading(false);
    }
  };

  // Handle country click to show history
  const handleCountryClick = (country) => {
    setSelectedCountry(country);

    // Extract historical data for this country from our existing data
    const history = [];

    // Go through all years in our data
    Object.keys(expenditureData).forEach((year) => {
      const yearData = expenditureData[year];
      if (yearData[country]) {
        history.push({
          Year: parseInt(year),
          Expenditure: yearData[country],
        });
      }
    });

    // Sort by year
    history.sort((a, b) => a.Year - b.Year);

    setCountryHistoryData(history);
  };

  // Close the history panel
  const closeHistoryPanel = () => {
    setSelectedCountry(null);
    setCountryHistoryData([]);
  };

  // Calculate circle radius based on expenditure using improved scaling
  const getCircleRadius = (expenditure) => {
    if (!expenditure) return 0;

    // Using square root scaling to make area proportional to value
    // This gives a more accurate visual representation
    const minRadius = 5;
    const maxRadius = 30;

    // Square root creates a better visual scale for the bubbles
    const normalizedValue = Math.sqrt(expenditure / maxExpenditure);
    const radius = minRadius + normalizedValue * (maxRadius - minRadius);

    return radius;
  };

  // Get country center point from GeoJSON
  const getCountryCenter = (feature) => {
    if (!feature || !feature.geometry) return [0, 0];

    if (feature.geometry.type === "Polygon") {
      // Calculate the center of the first polygon
      const coordinates = feature.geometry.coordinates[0];
      const latSum = coordinates.reduce((sum, coord) => sum + coord[1], 0);
      const lngSum = coordinates.reduce((sum, coord) => sum + coord[0], 0);
      return [latSum / coordinates.length, lngSum / coordinates.length];
    } else if (feature.geometry.type === "MultiPolygon") {
      // Use the first polygon in the multipolygon
      const coordinates = feature.geometry.coordinates[0][0];
      const latSum = coordinates.reduce((sum, coord) => sum + coord[1], 0);
      const lngSum = coordinates.reduce((sum, coord) => sum + coord[0], 0);
      return [latSum / coordinates.length, lngSum / coordinates.length];
    }

    return [0, 0];
  };

  // Get color for bubbles based on expenditure amount
  const getBubbleColor = (expenditure) => {
    if (!expenditure) return "#cccccc";

    // Color scale from yellow to red based on percentage of max value
    const ratio = Math.min(expenditure / maxExpenditure, 1);

    // Create color from green to red
    if (ratio < 0.25) {
      return "#4daf4a"; // Green for low values
    } else if (ratio < 0.5) {
      return "#ffff33"; // Yellow for medium values
    } else if (ratio < 0.75) {
      return "#ff7f00"; // Orange for high values
    } else {
      return "#e41a1c"; // Red for very high values
    }
  };

  if (loading) return <div>Loading map data...</div>;
  if (error) return <div>Error: {error}</div>;

  const yearExpenditures = expenditureData[selectedYear] || {};

  // Create sample sizes for legend
  const legendSizes = [
    Math.round(maxExpenditure * 0.01),
    Math.round(maxExpenditure * 0.05),
    Math.round(maxExpenditure * 0.25),
    Math.round(maxExpenditure * 0.5),
    Math.round(maxExpenditure * 0.75),
  ].filter((size) => size > 0);

  // Format expenditure value based on data mode
  const formatExpenditureValue = (value) => {
    if (dataMode === 'gdp') {
      return `${value.toFixed(2)}% of GDP`;
    } else {
      return `$${value.toLocaleString()} million`;
    }
  };

  return (
    <div
      className="expenditure-map-container"
      style={{ height: "100vh", width: "100vw", position: "relative" }}
    >
      <MapContainer
        center={[20, 0]}
        zoom={2}
        style={{ height: "100%", width: "100%" }}
        minZoom={1}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />

        <LayerGroup>
          {geoData &&
            geoData.features.map((feature) => {
              const countryName = feature.properties.Country;
              const expenditure = yearExpenditures[countryName];

              if (!countryName) return null;
              if (!expenditure) return null;

              const center =
                feature.properties.centroid || getCountryCenter(feature);
              const radius = getCircleRadius(expenditure);

              // Don't render if radius is 0 (no data)
              if (radius === 0) return null;

              return (
                <CircleMarker
                  key={`bubble-${countryName}`}
                  center={center}
                  radius={radius}
                  fillColor={getBubbleColor(expenditure)}
                  color="#000"
                  weight={1}
                  opacity={0.9}
                  fillOpacity={0.7}
                  // eventHandlers={{
                  //   click: () => handleCountryClick(countryName),
                  // }}
                >
                  <Popup>
                    <div>
                      <strong>{countryName}</strong>
                      <br />
                      Year: {selectedYear}
                      <br />
                      Military Expenditure: {formatExpenditureValue(expenditure)}
                      <br />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCountryClick(countryName);
                        }}
                        style={{
                          marginTop: "10px",
                          padding: "5px 10px",
                          backgroundColor: "#4285f4",
                          color: "white",
                          border: "none",
                          borderRadius: "4px",
                          cursor: "pointer",
                        }}
                      >
                        View History
                      </button>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
        </LayerGroup>
      </MapContainer>

      {/* Year slider control */}
      <div
        style={{
          position: "absolute",
          bottom: "20px",
          left: "50%",
          transform: "translateX(-50%)",
          background: "white",
          padding: "10px",
          borderRadius: "5px",
          boxShadow: "0 0 10px rgba(0,0,0,0.2)",
          zIndex: 1000,
          width: "80%",
          maxWidth: "500px",
        }}
      >
        <div style={{ marginBottom: "5px", textAlign: "center" }}>
          Year: {selectedYear}
        </div>
        <input
          type="range"
          min="1988"
          max="2022"
          value={selectedYear}
          onChange={(e) => setSelectedYear(parseInt(e.target.value))}
          style={{ width: "100%" }}
        />
      </div>

      {/* Legend */}
      <div
        style={{
          position: "absolute",
          top: "20px",
          right: "20px",
          background: "white",
          padding: "10px",
          borderRadius: "5px",
          boxShadow: "0 0 10px rgba(0,0,0,0.2)",
          zIndex: 1000,
        }}
      >
        <div style={{ fontWeight: "bold", marginBottom: "5px" }}>
          {dataMode === 'gdp' ? 'Military Expenditure (% of GDP)' : 'Military Expenditure ($ millions)'}
        </div>
        {legendSizes.map((size) => (
          <div
            key={`legend-${size}`}
            style={{ display: "flex", alignItems: "center", margin: "5px 0" }}
          >
            <div
              style={{
                width: `${getCircleRadius(size) * 2}px`,
                height: `${getCircleRadius(size) * 2}px`,
                borderRadius: "50%",
                background: getBubbleColor(size),
                marginRight: "10px",
              }}
            ></div>
            <div>{formatExpenditureValue(size)}</div>
          </div>
        ))}
      </div>

      {/* Country history panel */}
      {selectedCountry && (
        <div
          style={{
            position: "absolute",
            top: "20px",
            left: "20px",
            background: "white",
            padding: "15px",
            borderRadius: "5px",
            boxShadow: "0 0 10px rgba(0,0,0,0.2)",
            zIndex: 1000,
            maxWidth: "500px",
            maxHeight: "80vh",
            overflow: "auto",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "15px",
            }}
          >
            <h3 style={{ margin: 0 }}>{selectedCountry} - Historical Data</h3>
            <button
              onClick={closeHistoryPanel}
              style={{
                background: "none",
                border: "none",
                fontSize: "20px",
                cursor: "pointer",
              }}
            >
              ×
            </button>
          </div>
          <ChartComponent
            timeSeriesData={countryHistoryData}
            selectedCountry={selectedCountry}
            dataMode={dataMode}
          />
        </div>
      )}
    </div>
  );
};

export default ExpenditureMapComponent;
