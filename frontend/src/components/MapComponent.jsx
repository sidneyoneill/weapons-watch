// src/components/MapComponent.jsx
import React, { useEffect, useState } from 'react';
// Import Leaflet CSS first
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import axios from 'axios';
// Add this import for the Leaflet icon fix
import L from 'leaflet';
import { API_URL } from '../config'; // Add this import

// Fix Leaflet's default icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const MapComponent = ({ onCountrySelect }) => {
  const [geoData, setGeoData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    axios.get(`${API_URL}/geo_data`)
      .then(response => {
        try {
          const parsedData = JSON.parse(response.data.data);
          setGeoData(parsedData);
          setLoading(false);
        } catch (e) {
          console.error("Error parsing geo data:", e);
          setError("Failed to parse geo data");
          setLoading(false);
        }
      })
      .catch(error => {
        console.error("Error fetching geo data:", error);
        setError("Failed to fetch geo data");
        setLoading(false);
      });
  }, []);

  const countryStyle = {
    fillColor: '#1a83a1',
    weight: 1,
    opacity: 1,
    color: 'white',
    dashArray: '3',
    fillOpacity: 0.7
  };

  const onEachFeature = (feature, layer) => {
    const countryName = feature.properties.Country || feature.properties.name;
    layer.bindPopup(countryName);
    layer.on({
      click: () => {
        if (onCountrySelect) {
          onCountrySelect(countryName);
        }
      },
      mouseover: (e) => {
        const layer = e.target;
        layer.setStyle({
          weight: 2,
          color: '#666',
          dashArray: '',
          fillOpacity: 0.9
        });
      },
      mouseout: (e) => {
        const layer = e.target;
        layer.setStyle(countryStyle);
      }
    });
  };

  if (loading) return <div>Loading map data...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="map-container">
      <h2>Trade Flow Map</h2>
      <div style={{ height: "500px", width: "100%" }}>
        <MapContainer 
          center={[20, 0]} 
          zoom={2} 
          style={{ height: "100%", width: "100%" }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          {geoData && (
            <GeoJSON 
              data={geoData} 
              style={countryStyle}
              onEachFeature={onEachFeature}
            />
          )}
        </MapContainer>
      </div>
    </div>
  );
};

export default MapComponent;
