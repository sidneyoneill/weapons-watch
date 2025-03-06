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

const ExpenditureMapComponent = () => {
  const [geoData, setGeoData] = useState(null);
  const [expenditureData, setExpenditureData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedYear, setSelectedYear] = useState(2000); // Default year
  const [maxExpenditure, setMaxExpenditure] = useState(1); // For scaling circles
  const [selectedCountry, setSelectedCountry] = useState(null); // Track clicked country
  const [countryHistoryData, setCountryHistoryData] = useState([]); // Store historical data

  // Fetch geo data on mount
  useEffect(() => {
    setLoading(true);
    axios
      .get("http://localhost:8000/geo_data")
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
  }, []);

  // Fetch expenditure data for all countries
  const fetchAllExpenditureData = async () => {
    try {
      const response = await axios.get(
        "http://localhost:8000/all_expenditures"
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
                      Military Expenditure: ${expenditure.toLocaleString()}{" "}
                      million
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
                        View Historical Data
                      </button>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
        </LayerGroup>
      </MapContainer>

      {/* Year slider overlay */}
      <div
        style={{
          position: "absolute",
          bottom: "20px",
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(255,255,255,0.9)",
          padding: "15px",
          borderRadius: "5px",
          width: "80%",
          maxWidth: "800px",
          textAlign: "center",
          zIndex: 1000,
          boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
        }}
      >
        <div style={{ marginBottom: "10px" }}>
          <span style={{ fontSize: "18px", fontWeight: "bold" }}>
            Military Expenditure - Year: {selectedYear}
          </span>
        </div>
        <input
          type="range"
          min={1968}
          max={2024}
          value={selectedYear}
          onChange={(e) => setSelectedYear(parseInt(e.target.value))}
          style={{ width: "100%" }}
        />
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span>1968</span>
          <span>2024</span>
        </div>
      </div>

      {/* Bubble Legend */}
      <div
        style={{
          position: "absolute",
          top: "20px",
          right: "20px",
          background: "rgba(255,255,255,0.9)",
          padding: "15px",
          borderRadius: "5px",
          zIndex: 1000,
          boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
        }}
      >
        <div style={{ marginBottom: "10px", fontWeight: "bold" }}>
          Military Expenditure ($ million)
        </div>

        {legendSizes.map((size) => (
          <div
            key={size}
            style={{
              display: "flex",
              alignItems: "center",
              marginBottom: "8px",
            }}
          >
            <div
              style={{
                width: getCircleRadius(size) * 2,
                height: getCircleRadius(size) * 2,
                borderRadius: "50%",
                background: getBubbleColor(size),
                marginRight: "10px",
                border: "1px solid black",
              }}
            ></div>
            <span>{size.toLocaleString()}</span>
          </div>
        ))}
      </div>

      {/* Country History Panel */}
      {selectedCountry && countryHistoryData.length > 0 && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            background: "rgba(255,255,255,0.95)",
            padding: "20px",
            borderRadius: "8px",
            zIndex: 2000,
            boxShadow: "0 5px 15px rgba(0,0,0,0.3)",
            width: "80%",
            maxWidth: "800px",
            maxHeight: "80vh",
            overflow: "auto",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: "15px",
            }}
          >
            <h2 style={{ margin: 0 }}></h2>
            <button
              onClick={closeHistoryPanel}
              style={{
                background: "transparent",
                border: "none",
                fontSize: "1.5rem",
                cursor: "pointer",
                padding: "0 5px",
              }}
            >
              &times;
            </button>
          </div>

          <div style={{ height: "400px" }}>
            <ChartComponent
              timeSeriesData={countryHistoryData}
              selectedCountry={selectedCountry}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ExpenditureMapComponent;
