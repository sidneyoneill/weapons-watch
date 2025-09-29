import React, { useEffect, useState, useRef } from "react";
import Globe from "react-globe.gl";
import axios from "axios";
import ChartComponent from "./ChartComponent";
import * as THREE from 'three';

const GlobeComponent = ({ dataMode = 'total' }) => {
  // All state hooks need to be declared at the top level
  const [geoData, setGeoData] = useState(null);
  const [expenditureData, setExpenditureData] = useState({});
  const [gdpData, setGdpData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedYear, setSelectedYear] = useState(2000); // Default year
  const [maxExpenditure, setMaxExpenditure] = useState(1); // For scaling
  const [maxGdpPercentage, setMaxGdpPercentage] = useState(1); // For GDP percentage scaling
  const [selectedCountry, setSelectedCountry] = useState(null); // Track clicked country
  const [countryHistoryData, setCountryHistoryData] = useState([]); // Store historical data
  const [altitude, setAltitude] = useState(1.5); // Adjust zoom level
  // Fixed visualization settings that cannot be changed by user
  const [visualizationMode, setVisualizationMode] = useState('country-color'); // Changed to country-color mode
  const [showLabels, setShowLabels] = useState(false); // Toggle for country labels
  const [topCountries, setTopCountries] = useState([]);
  const [colorScheme, setColorScheme] = useState('orangeHeat'); // 'viridis', 'inferno', 'plasma', 'magma', 'greens', 'emerald'
  const [ringsData, setRingsData] = useState([]);
  const [pulsesData, setPulsesData] = useState([]);
  const [pointsData, setPointsData] = useState([]);
  const [dotSize, setDotSize] = useState(1.2); // Fixed default dot size
  const [arcsData, setArcsData] = useState([]);
  const [labelData, setLabelData] = useState([]);
  // New state for top trade relationships
  const [topTradeRelationships, setTopTradeRelationships] = useState([]);
  // State for trade relationships
  const [tradePartners, setTradePartners] = useState([]);
  // New state for trade view mode: 'all', 'import', 'export'
  const [tradeViewMode, setTradeViewMode] = useState('all');
  // New state for country color data
  const [countryColors, setCountryColors] = useState({});
  
  // All refs after state hooks
  const globeRef = useRef();
  
  // Function to generate label data for specific countries or all if none specified
  const generateLabelData = (countryFilter = null) => {
    if (!geoData) return [];
    
    // Filter features based on country filter if provided
    const filteredFeatures = countryFilter 
      ? geoData.features.filter(feature => 
          feature.properties && 
          feature.properties.Country && 
          countryFilter.includes(feature.properties.Country)
        )
      : geoData.features.filter(feature => 
          feature.properties && 
          feature.properties.Country
        );
    
    return filteredFeatures
      .map(feature => {
        // Calculate centroid properly for each country
        let lat = 0, lng = 0;
        let coordinateCount = 0;
        
        try {
          if (!feature.geometry) return null;
          
          if (feature.geometry.type === "Polygon") {
            // Use the first polygon's coordinates
            const coordinates = feature.geometry.coordinates[0];
            coordinates.forEach(coord => {
              lng += coord[0];
              lat += coord[1];
              coordinateCount++;
            });
          } else if (feature.geometry.type === "MultiPolygon") {
            // Use the first polygon in the multipolygon
            const coordinates = feature.geometry.coordinates[0][0];
            coordinates.forEach(coord => {
              lng += coord[0];
              lat += coord[1];
              coordinateCount++;
            });
          }
          
          if (coordinateCount > 0) {
            lat = lat / coordinateCount;
            lng = lng / coordinateCount;
            
            if (isNaN(lat) || isNaN(lng)) {
              console.log(`Invalid calculated coordinates for ${feature.properties.Country}`);
              return null;
            }
          }
        } catch (error) {
          console.error(`Error calculating centroid for ${feature.properties.Country}:`, error);
          return null;
        }
        
        // Highlight selected country label if applicable
        const isSelected = selectedCountry === feature.properties.Country;
        
        return {
          lat: lat,
          lng: lng,
          text: feature.properties.Country,
          size: isSelected ? 0.8 : 0.5, // Make selected country label larger
          country: feature.properties.Country,
          color: isSelected ? '#ff7b00' : '#ffffff' // Highlight selected country
        };
      })
      .filter(label => label !== null);
  };

  // All useEffect hooks
  // Effect 1: Fetch geo and expenditure data
  useEffect(() => {
    setLoading(true);
    
    // Fetch geo data
    axios.get(`http://localhost:8000/geo_data`)
      .then((response) => {
        try {
          const parsedData = JSON.parse(response.data.data);
          setGeoData(parsedData);
          
          // Once geo data is loaded, fetch both expenditure datasets
          fetchExpenditureData('total');
          fetchExpenditureData('gdp');
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

  // Effect 2: Initialize visualization once all data is loaded
  useEffect(() => {
    // Only proceed when both data and geo information are loaded
    if (!loading && geoData && Object.keys(expenditureData).length > 0) {
      console.log("All data loaded, initializing visualization");
      if (visualizationMode === 'country-color') {
        generateCountryColors();
      } else if (visualizationMode === 'dots') {
        generatePointsData();
      }
    }
  }, [loading, geoData, expenditureData, gdpData]);

  // Effect 3: Update data when year or dataMode changes
  useEffect(() => {
    if (!loading && geoData && Object.keys(expenditureData).length > 0) {
      console.log("Year or dataMode changed, updating visualization");
      if (visualizationMode === 'country-color') {
        generateCountryColors();
      } else if (visualizationMode === 'dots') {
        generatePointsData();
      } else if (visualizationMode === 'pulse' || visualizationMode === 'rings') {
        generateVisualizationData(visualizationMode);
      }
      
      // Clear arcs when year changes to avoid showing outdated connections
      setArcsData([]);
      
      // If a country is selected, also update its historical data
      if (selectedCountry) {
        const history = [];
        const dataSource = dataMode === 'gdp' ? gdpData : expenditureData;
        
        // Re-extract historical data for the selected country with the new dataMode
        Object.keys(dataSource).forEach((year) => {
          const yearData = dataSource[year];
          if (yearData[selectedCountry]) {
            history.push({
              Year: parseInt(year),
              Expenditure: yearData[selectedCountry],
            });
          }
        });
        
        // Sort by year
        history.sort((a, b) => a.Year - b.Year);
        setCountryHistoryData(history);
      }
    }
  }, [selectedYear, dataMode]);

  // Effect 4: Update label data when showLabels changes
  useEffect(() => {
    if (showLabels) {
      if (selectedCountry && arcsData.length > 0) {
        // If a country is selected, only show that country and trade partners
        const partnerCountries = arcsData.map(arc => arc.tradePartner);
        setLabelData(generateLabelData([selectedCountry, ...partnerCountries]));
      } else {
        // Otherwise show all countries
        setLabelData(generateLabelData());
      }
    } else {
      setLabelData([]);
    }
  }, [showLabels, selectedCountry, arcsData]);

  // Add resize handler to ensure proper sizing when window is resized
  useEffect(() => {
    const handleResize = () => {
      if (globeRef.current) {
        // Force the globe to update its dimensions
        globeRef.current.width = window.innerWidth;
        globeRef.current.height = window.innerHeight;
      }
    };

    window.addEventListener('resize', handleResize);
    // Initial call to set correct size
    handleResize();
    
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Add force resize handler to ensure globe always takes full width
  useEffect(() => {
    const handleResize = () => {
      if (globeRef.current) {
        // Force the globe to update its dimensions
        const width = window.innerWidth;
        const height = window.innerHeight;
        globeRef.current.width = width;
        globeRef.current.height = height;
        // Force re-render by calling the component's internal resize
        if (globeRef.current._onResize) {
          globeRef.current._onResize(width, height);
        }
      }
    };

    // Apply immediately
    handleResize();
    
    // Add resize listener
    window.addEventListener('resize', handleResize);
    
    // Also set a periodic check to ensure it stays full width
    const intervalCheck = setInterval(handleResize, 1000);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      clearInterval(intervalCheck);
    };
  }, []);

  // Effect to update arcs when tradeViewMode changes
  useEffect(() => {
    if (selectedCountry && tradePartners.length > 0) {
      updateTradeArcs();
    }
  }, [tradeViewMode]);

  // Fetch expenditure data for all countries
  const fetchExpenditureData = async (mode) => {
    try {
      const response = await axios.get(
        `http://localhost:8000/all_expenditures?mode=${mode}`
      );
      const data = response.data.time_series;
      console.log(`Received ${mode} expenditure data:`, data.length, "records");
      
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

      if (mode === 'total') {
        setExpenditureData(organizedData);
        setMaxExpenditure(maxValue);
        
        // Find top countries by expenditure in the latest available year
        const latestYear = Math.max(...Object.keys(organizedData).map(Number));
        const latestYearData = organizedData[latestYear] || {};
        
        const topCountriesList = Object.entries(latestYearData)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 10)
          .map(([country, value]) => ({ country, value }));
        
        setTopCountries(topCountriesList);
      } else {
        setGdpData(organizedData);
        setMaxGdpPercentage(maxValue);
      }
      
      setLoading(false);
      
      // Don't try to generate points here - we'll do it in the useEffect below
    } catch (error) {
      console.error(`Error fetching ${mode} expenditure data:`, error);
      setLoading(false);
    }
  };

  // NEW FUNCTION: Generate country colors based on military expenditure
  const generateCountryColors = () => {
    if (!geoData) {
      console.warn("Cannot generate country colors: geoData is null");
      return;
    }
    
    const dataSource = dataMode === 'gdp' ? gdpData : expenditureData;
    const currentYearData = dataSource[selectedYear] || {};
    
    console.log("Generating country colors for year:", selectedYear);
    console.log("Countries with data:", Object.keys(currentYearData).length);
    
    if (Object.keys(currentYearData).length === 0) {
      console.warn("No data available for selected year:", selectedYear);
      setCountryColors({});
      return;
    }
    
    const maxValue = dataMode === 'gdp' ? maxGdpPercentage : maxExpenditure;
    
    // Find all values for percentile calculations
    const allValues = [];
    geoData.features.forEach(feature => {
      if (!feature.properties || !feature.properties.Country) return;
      
      const countryName = feature.properties.Country;
      const value = currentYearData[countryName];
      
      if (value) {
        allValues.push(value);
      }
    });
    
    // Sort values for percentile calculations
    allValues.sort((a, b) => a - b);
    
    // Helper function to find percentile
    const getPercentile = (arr, p) => {
      const index = Math.floor(arr.length * p);
      return arr[index];
    };
    
    // Calculate percentile thresholds
    const p25 = getPercentile(allValues, 0.25);
    const p50 = getPercentile(allValues, 0.50);
    const p75 = getPercentile(allValues, 0.75);
    const p90 = getPercentile(allValues, 0.90);
    const p95 = getPercentile(allValues, 0.95);
    
    console.log("Value distribution:", {
      min: allValues[0],
      p25,
      p50,
      p75,
      p90,
      p95,
      max: allValues[allValues.length - 1]
    });
    
    // Create color mapping for each country
    const colors = {};
    
    geoData.features.forEach(feature => {
      if (!feature.properties || !feature.properties.Country) return;
      
      const countryName = feature.properties.Country;
      const value = currentYearData[countryName];
      
      if (!value) {
        // No data, use dark gray
        colors[countryName] = "#333333";
        return;
      }
      
      // Determine color category based on percentile
      let colorCategory;
      if (value >= p95) colorCategory = 5;       // Top 5% - darkest/most intense
      else if (value >= p90) colorCategory = 4;  // 90-95 percentile
      else if (value >= p75) colorCategory = 3;  // 75-90 percentile
      else if (value >= p50) colorCategory = 2;  // 50-75 percentile
      else if (value >= p25) colorCategory = 1;  // 25-50 percentile
      else colorCategory = 0;                    // Bottom 25% - lightest
      
      colors[countryName] = getCountryColorByCategory(colorCategory);
    });
    
    setCountryColors(colors);
  };

  // Generate points data for visualization on the globe
  const generatePointsData = () => {
    if (!geoData) {
      console.warn("Cannot generate points: geoData is null");
      return;
    }
    
    const dataSource = dataMode === 'gdp' ? gdpData : expenditureData;
    const currentYearData = dataSource[selectedYear] || {};
    
    console.log("Generating points for year:", selectedYear);
    console.log("Countries with data:", Object.keys(currentYearData).length);
    
    if (Object.keys(currentYearData).length === 0) {
      console.warn("No data available for selected year:", selectedYear);
      setPointsData([]);
      return;
    }
    
    const maxValue = dataMode === 'gdp' ? maxGdpPercentage : maxExpenditure;
    
    // Debug which features have geometry and properties
    const featuresWithProperties = geoData.features.filter(feature => 
      feature.properties && feature.properties.Country
    ).length;
    
    const featuresWithGeometry = geoData.features.filter(feature => 
      feature.geometry && 
      (feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon')
    ).length;
    
    console.log("Features with properties:", featuresWithProperties);
    console.log("Features with geometry:", featuresWithGeometry);
    
    // Find quartile values for better scaling
    const allValues = [];
    geoData.features.forEach(feature => {
      if (!feature.properties || !feature.properties.Country) return;
      
      const countryName = feature.properties.Country;
      const value = currentYearData[countryName];
      
      if (value) {
        allValues.push(value);
      }
    });
    
    // Sort values for percentile calculations
    allValues.sort((a, b) => a - b);
    
    // Helper function to find percentile
    const getPercentile = (arr, p) => {
      const index = Math.floor(arr.length * p);
      return arr[index];
    };
    
    // Calculate percentile thresholds
    const p25 = getPercentile(allValues, 0.25);
    const p50 = getPercentile(allValues, 0.50);
    const p75 = getPercentile(allValues, 0.75);
    const p90 = getPercentile(allValues, 0.90);
    const p95 = getPercentile(allValues, 0.95);
    
    console.log("Value distribution:", {
      min: allValues[0],
      p25,
      p50,
      p75,
      p90,
      p95,
      max: allValues[allValues.length - 1]
    });
    
    // Get country data with coordinates calculated from geometry
    const points = [];
    
    geoData.features.forEach(feature => {
      if (!feature.properties || !feature.properties.Country) return;
      
      const countryName = feature.properties.Country;
      let value = currentYearData[countryName];
      
      // Special case for United States to handle different naming conventions
      if (countryName === "United States" && !value) {
        value = currentYearData["United States of America"];
      } else if (countryName === "United States of America" && !value) {
        value = currentYearData["United States"];
      }
      
      if (!value || !feature.geometry) return;
      
      // Calculate centroid from geometry
      let lat = 0, lng = 0;
      let coordinateCount = 0;
      
      try {
        if (feature.geometry.type === "Polygon") {
          // Use the first polygon's coordinates
          const coordinates = feature.geometry.coordinates[0];
          coordinates.forEach(coord => {
            lng += coord[0];
            lat += coord[1];
            coordinateCount++;
          });
        } else if (feature.geometry.type === "MultiPolygon") {
          // Use the first polygon in the multipolygon
          const coordinates = feature.geometry.coordinates[0][0];
          coordinates.forEach(coord => {
            lng += coord[0];
            lat += coord[1];
            coordinateCount++;
          });
        }
        
        if (coordinateCount > 0) {
          lat = lat / coordinateCount;
          lng = lng / coordinateCount;
          
          if (isNaN(lat) || isNaN(lng)) {
            console.log(`Invalid calculated coordinates for ${countryName}`);
            return;
          }
          
          // DRAMATICALLY improved scaling to create more visible differences between countries
          // Use percentile-based sizing to make top countries stand out more dramatically
          let sizeScaleFactor;
          
          if (value >= p95) {
            // Top 5% countries - largest dots (3.5x - 5x base size)
            const topRatio = (value - p95) / (maxValue - p95); // Position within top 5%
            sizeScaleFactor = 3.5 + (topRatio * 1.5); // Scale from 3.5x to 5x
          } else if (value >= p90) {
            // 90-95 percentile - large dots (2.5x - 3.5x)
            const topRatio = (value - p90) / (p95 - p90);
            sizeScaleFactor = 2.5 + (topRatio * 1.0);
          } else if (value >= p75) {
            // 75-90 percentile - medium-large dots (1.5x - 2.5x)
            const topRatio = (value - p75) / (p90 - p75);
            sizeScaleFactor = 1.5 + (topRatio * 1.0);
          } else if (value >= p50) {
            // 50-75 percentile - medium dots (1.0x - 1.5x)
            const topRatio = (value - p50) / (p75 - p50);
            sizeScaleFactor = 1.0 + (topRatio * 0.5);
          } else if (value >= p25) {
            // 25-50 percentile - small-medium dots (0.6x - 1.0x)
            const topRatio = (value - p25) / (p50 - p25);
            sizeScaleFactor = 0.6 + (topRatio * 0.4);
          } else {
            // Bottom 25% - smallest dots (0.3x - 0.6x)
            const topRatio = value / p25;
            sizeScaleFactor = 0.3 + (topRatio * 0.3);
          }
          
          // Calculate final size with the dramatically improved scaling
          const size = sizeScaleFactor * dotSize;
          
          // Determine color category based on percentile too
          let colorCategory;
          if (value >= p90) colorCategory = 5;       // Top 10% - darkest/most intense
          else if (value >= p75) colorCategory = 4;  // 75-90 percentile
          else if (value >= p50) colorCategory = 3;  // 50-75 percentile
          else if (value >= p25) colorCategory = 2;  // 25-50 percentile
          else colorCategory = 1;                    // Bottom 25% - lightest
          
          const color = getDotColorByCategory(colorCategory);
          
          points.push({
            lat: lat,
            lng: lng,
            size: size,
            color: color,
            country: countryName,
            value: value,
            sizeCategory: colorCategory, // Store category for debugging
            percentile: value >= p95 ? "Top 5%" : 
                       value >= p90 ? "90-95%" : 
                       value >= p75 ? "75-90%" : 
                       value >= p50 ? "50-75%" :
                       value >= p25 ? "25-50%" : "Bottom 25%"
          });
        }
      } catch (error) {
        console.error(`Error calculating centroid for ${countryName}:`, error);
      }
    });
    
    console.log("Generated points:", points.length);
    
    // Sort points so that smaller dots are rendered first and larger dots on top
    points.sort((a, b) => a.size - b.size);
    
    setPointsData(points);
  };

  // Generate ring or pulse data
  const generateVisualizationData = (mode) => {
    if (!geoData) {
      console.warn("Cannot generate visualization: geoData is null");
      return;
    }
    
    const dataSource = dataMode === 'gdp' ? gdpData : expenditureData;
    const yearData = dataSource[selectedYear] || {};
    
    if (Object.keys(yearData).length === 0) {
      console.warn("No data available for selected year:", selectedYear);
      mode === 'pulse' ? setPulsesData([]) : setRingsData([]);
      return;
    }
    
    const maxValue = dataMode === 'gdp' ? maxGdpPercentage : maxExpenditure;
    
    // Get top countries for visualization
    const topCountriesWithCoordinates = [];
    
    // First collect all countries with data
    const countriesWithValues = Object.entries(yearData)
      .filter(([country, value]) => value > 0)
      .sort(([countryA, valueA], [countryB, valueB]) => valueB - valueA)
      .slice(0, 25); // Get top 25 to ensure we find at least 15 with valid coordinates
    
    // Now calculate coordinates for each country
    countriesWithValues.forEach(([country, value]) => {
      // Find country feature in geoData
      const feature = geoData.features.find(f => f.properties.Country === country);
      if (!feature || !feature.geometry) return;
      
      try {
        // Calculate centroid from geometry (same as in generatePointsData)
        let lat = 0, lng = 0;
        let coordinateCount = 0;
        
        if (feature.geometry.type === "Polygon") {
          // Use the first polygon's coordinates
          const coordinates = feature.geometry.coordinates[0];
          coordinates.forEach(coord => {
            lng += coord[0];
            lat += coord[1];
            coordinateCount++;
          });
        } else if (feature.geometry.type === "MultiPolygon") {
          // Use the first polygon in the multipolygon
          const coordinates = feature.geometry.coordinates[0][0];
          coordinates.forEach(coord => {
            lng += coord[0];
            lat += coord[1];
            coordinateCount++;
          });
        }
        
        if (coordinateCount > 0) {
          lat = lat / coordinateCount;
          lng = lng / coordinateCount;
          
          if (isNaN(lat) || isNaN(lng)) {
            console.log(`Invalid calculated coordinates for ${country}`);
            return;
          }
          
          const ratio = Math.min(value / maxValue, 1);
          
          // Apply enhanced, non-linear scaling for better visualization
          // Use power scaling for more pronounced differences
          const powerScale = Math.pow(ratio, 0.5);
          
          const color = getDotColor(ratio);
          
          topCountriesWithCoordinates.push({
            lat,
            lng,
            country,
            value,
            ratio: powerScale, // Use enhanced scaling
            color
          });
        }
      } catch (error) {
        console.error(`Error calculating centroid for ${country}:`, error);
      }
    });
    
    // Limit to top 15 for performance
    const topCountries = topCountriesWithCoordinates.slice(0, 15);
    
    console.log(`Generated ${mode} data for top countries:`, topCountries.length);
    
    if (mode === 'pulse') {
      setPulsesData(topCountries.map(country => ({
        lat: country.lat,
        lng: country.lng,
        color: country.color,
        maxR: 3 + 25 * country.ratio, // Increased range for more visibility
        propagationSpeed: 1.5,
        repeatPeriod: 1000 + 2000 * (1 - country.ratio) // Faster pulses for higher values
      })));
    } else if (mode === 'rings') {
      setRingsData(topCountries.map(country => ({
        lat: country.lat,
        lng: country.lng,
        label: country.country,
        altitude: 0,
        radius: 0.8 + 2.0 * country.ratio, // Increased scaling factor for more visible differences
        color: country.color,
        value: country.value
      })));
    }
  };

  // Get coordinates for a country by name
  const getCountryCoordinates = (countryName) => {
    if (!geoData) return null;
    
    const feature = geoData.features.find(f => 
      f.properties && f.properties.Country === countryName
    );
    
    if (!feature || !feature.geometry) return null;
    
    let lat = 0, lng = 0;
    let coordinateCount = 0;
    
    try {
      if (feature.geometry.type === "Polygon") {
        // Use the first polygon's coordinates
        const coordinates = feature.geometry.coordinates[0];
        coordinates.forEach(coord => {
          lng += coord[0];
          lat += coord[1];
          coordinateCount++;
        });
      } else if (feature.geometry.type === "MultiPolygon") {
        // Use the first polygon in the multipolygon
        const coordinates = feature.geometry.coordinates[0][0];
        coordinates.forEach(coord => {
          lng += coord[0];
          lat += coord[1];
          coordinateCount++;
        });
      }
      
      if (coordinateCount > 0) {
        lat = lat / coordinateCount;
        lng = lng / coordinateCount;
        
        if (isNaN(lat) || isNaN(lng)) {
          console.log(`Invalid calculated coordinates for ${countryName}`);
          return null;
        }
        
        return [lat, lng];
      }
    } catch (error) {
      console.error(`Error calculating centroid for ${countryName}:`, error);
    }
    
    return null;
  };

  // Handle country click to show history
  const handleCountryClick = async (country) => {
    const countryName = country.country || country.properties?.Country;
    if (!countryName) return;
    
    setSelectedCountry(countryName);
    console.log(`Country selected: ${countryName}`);

    // Center the globe view on the selected country
    if (globeRef.current) {
      let lat, lng;
      
      if (country.lat && country.lng) {
        lat = country.lat;
        lng = country.lng;
      } else if (country.properties?.centroid) {
        [lng, lat] = country.properties.centroid;
      }
      
      if (lat && lng) {
        globeRef.current.pointOfView({ lat, lng, altitude }, 1000);
      }
    }

    // Extract historical data for this country
    const history = [];
    const dataSource = dataMode === 'gdp' ? gdpData : expenditureData;
    
    // Go through all years in our data
    Object.keys(dataSource).forEach((year) => {
      const yearData = dataSource[year];
      if (yearData[countryName]) {
        history.push({
          Year: parseInt(year),
          Expenditure: yearData[countryName],
        });
      }
    });
    
    // Sort by year
    history.sort((a, b) => a.Year - b.Year);
    setCountryHistoryData(history);

    try {
      console.log(`Fetching trade partners for: ${countryName}`);
      const response = await axios.get(
        `http://localhost:8000/trade_partners/${encodeURIComponent(countryName)}`
      );
      
      console.log("API response:", response);
      
      // Check if the response has data
      if (!response.data || response.data.length === 0) {
        console.log(`No trade partner data returned for ${countryName}`);
        setArcsData([]);
        setTopTradeRelationships([]);
        return;
      }
      
      const tradeData = response.data;
      console.log(`Received ${tradeData.length} trade partners for ${countryName}`);
      
      // Process the trade relationships to get top 5
      const processedTradeData = tradeData
        .filter(partner => partner.country !== countryName) // Exclude self-references
        .sort((a, b) => b.value - a.value) // Sort by value (descending)
        .slice(0, 5); // Take top 5
      
      setTopTradeRelationships(processedTradeData);

      // Convert trade data into arc format
      const srcCoords = getCountryCoordinates(countryName);
      if (!srcCoords) {
        console.warn(`Could not find coordinates for source country: ${countryName}`);
        return;
      }

      // Try with both "United States" and "United States of America" for better matching
      const alternativeNames = {
        "United States": "United States of America",
        "United States of America": "United States",
        "USA": "United States",
        "Russia": "Soviet Union",
        "Soviet Union": "Russia"
      };

      // First set empty arcs to clear any existing ones
      setArcsData([]);
      
      // Then after a short delay, add the new arcs to create the animation effect
      setTimeout(() => {
        const newArcs = tradeData
          .filter(partner => partner.country !== countryName) // Exclude self-references
          .map(partner => {
            let targetCoords = getCountryCoordinates(partner.country);
            
            // If not found, try alternative name
            if (!targetCoords && alternativeNames[partner.country]) {
              console.log(`Trying alternative name: ${alternativeNames[partner.country]}`);
              targetCoords = getCountryCoordinates(alternativeNames[partner.country]);
            }
            
            if (!targetCoords) {
              console.warn(`Could not find coordinates for trade partner: ${partner.country}`);
              return null;
            }
            
            // Brighter orange color with higher base opacity
            const opacity = Math.min(0.5 + partner.value / 10000, 0.95);
            const lineColor = `rgba(249, 115, 22, ${opacity})`; // Brighter orange color
            
            return {
              startLat: srcCoords[0],
              startLng: srcCoords[1],
              endLat: targetCoords[0],
              endLng: targetCoords[1],
              color: lineColor,
              value: partner.value,
              tradePartner: partner.country,
              tradeValue: partner.value,
              tradeType: partner.type
            };
          })
          .filter(arc => arc !== null);

        console.log(`Generated ${newArcs.length} trade arcs for ${countryName}`);
        setArcsData(newArcs);
        
        // Update the labels to show only selected country and partners
        if (showLabels) {
          const partnerCountries = newArcs.map(arc => arc.tradePartner);
          const filteredLabelData = generateLabelData([countryName, ...partnerCountries]);
          setLabelData(filteredLabelData);
        }
      }, 100);
      
      // Animate to the selected country with a slightly higher altitude to see connections
      if (globeRef.current) {
        // Use srcCoords directly since we've already validated it exists
        globeRef.current.pointOfView({ 
          lat: srcCoords[0], 
          lng: srcCoords[1], 
          altitude: 2.2 
        }, 1000);
      }
    } catch (error) {
      console.error('Error fetching trade partners:', error);
      if (error.response) {
        console.error('API error response:', error.response.data);
      }
      setArcsData([]); // Clear previous arcs if there's an error
      setTopTradeRelationships([]); // Clear top trade relationships if there's an error
    }

    try {
      console.log(`Fetching trade partners for: ${countryName}`);
      const response = await axios.get(
        `http://localhost:8000/trade_partners/${encodeURIComponent(countryName)}`
      );
      
      console.log("API response:", response);
      
      // Check if the response has data
      if (!response.data || response.data.length === 0) {
        console.log(`No trade partner data returned for ${countryName}`);
        setArcsData([]);
        setTradePartners([]);
        return;
      }
      
      const tradeData = response.data;
      console.log(`Received ${tradeData.length} trade partners for ${countryName}`);
      
      // Process and organize the trade data by partner country
      const partnerMap = new Map();
      
      tradeData.forEach(partner => {
        if (partner.country === countryName) return; // Skip self-references
        
        if (!partnerMap.has(partner.country)) {
          partnerMap.set(partner.country, {
            country: partner.country,
            imports: 0,
            exports: 0,
            total: 0
          });
        }
        
        const currentData = partnerMap.get(partner.country);
        
        if (partner.type === 'import') {
          currentData.imports += partner.value;
        } else if (partner.type === 'export') {
          currentData.exports += partner.value;
        }
        
        currentData.total += partner.value;
        partnerMap.set(partner.country, currentData);
      });
      
      // Convert to array and sort by total trade value
      const processedTradeData = Array.from(partnerMap.values())
        .sort((a, b) => b.total - a.total)
        .slice(0, 5); // Take top 5
      
      setTradePartners(processedTradeData);

      // Convert trade data into arc format
      const srcCoords = getCountryCoordinates(countryName);
      if (!srcCoords) {
        console.warn(`Could not find coordinates for source country: ${countryName}`);
        return;
      }
      // Try with both "United States" and "United States of America" for better matching
      const alternativeNames = {
        "United States": "United States of America",
        "United States of America": "United States",
        "USA": "United States",
        "Russia": "Soviet Union",
        "Soviet Union": "Russia"
      };
      // First set empty arcs to clear any existing ones
      setArcsData([]);
      
      // Then after a short delay, add the new arcs to create the animation effect
      setTimeout(() => {
        const newArcs = tradeData
          .filter(partner => partner.country !== countryName) // Exclude self-references
          .map(partner => {
            let targetCoords = getCountryCoordinates(partner.country);
            
            // If not found, try alternative name
            if (!targetCoords && alternativeNames[partner.country]) {
              console.log(`Trying alternative name: ${alternativeNames[partner.country]}`);
              targetCoords = getCountryCoordinates(alternativeNames[partner.country]);
            }
            
            if (!targetCoords) {
              console.warn(`Could not find coordinates for trade partner: ${partner.country}`);
              return null;
            }
            
            // Brighter orange color with higher base opacity
            const opacity = Math.min(0.5 + partner.value / 10000, 0.95);
            const lineColor = `rgba(249, 115, 22, ${opacity})`; // Brighter orange color
            
            return {
              startLat: srcCoords[0],
              startLng: srcCoords[1],
              endLat: targetCoords[0],
              endLng: targetCoords[1],
              color: lineColor,
              value: partner.value,
              tradePartner: partner.country,
              tradeValue: partner.value,
              tradeType: partner.type
            };
          })
          .filter(arc => arc !== null);
        console.log(`Generated ${newArcs.length} trade arcs for ${countryName}`);
        setArcsData(newArcs);
        
        // Update the labels to show only selected country and partners
        if (showLabels) {
          const partnerCountries = newArcs.map(arc => arc.tradePartner);
          const filteredLabelData = generateLabelData([countryName, ...partnerCountries]);
          setLabelData(filteredLabelData);
        }
      }, 100);
      
      // Animate to the selected country with a slightly higher altitude to see connections
      if (globeRef.current) {
        // Use srcCoords directly since we've already validated it exists
        globeRef.current.pointOfView({ 
          lat: srcCoords[0], 
          lng: srcCoords[1], 
          altitude: 2.2 
        }, 1000);
      }
    } catch (error) {
      console.error('Error fetching trade partners:', error);
      if (error.response) {
        console.error('API error response:', error.response.data);
      }
      setArcsData([]); // Clear previous arcs if there's an error
      setTradePartners([]); // Clear trade partners if there's an error
    }
  };

  // Function to update trade arcs based on the trade view mode
  const updateTradeArcs = () => {
    const srcCoords = getCountryCoordinates(selectedCountry);
    if (!srcCoords) return;
    
    // First set empty arcs to clear any existing ones
    setArcsData([]);
    
    // Get all trade data for the country
    axios.get(`http://localhost:8000/trade_partners/${encodeURIComponent(selectedCountry)}`)
      .then(response => {
        if (!response.data || response.data.length === 0) return;
        
        const tradeData = response.data;
        
        // Filter based on tradeViewMode
        const filteredTradeData = tradeData.filter(partner => {
          if (tradeViewMode === 'all') return true;
          return partner.type === tradeViewMode;
        });
        
        // Process and create arcs with a short delay for animation effect
        setTimeout(() => {
          const alternativeNames = {
            "United States": "United States of America",
            "United States of America": "United States",
            "USA": "United States",
            "Russia": "Soviet Union",
            "Soviet Union": "Russia"
          };
          
          const newArcs = filteredTradeData
            .filter(partner => partner.country !== selectedCountry) // Exclude self-references
            .map(partner => {
              let targetCoords = getCountryCoordinates(partner.country);
              
              // If not found, try alternative name
              if (!targetCoords && alternativeNames[partner.country]) {
                console.log(`Trying alternative name: ${alternativeNames[partner.country]}`);
                targetCoords = getCountryCoordinates(alternativeNames[partner.country]);
              }
              
              if (!targetCoords) return null;
              
              // Colors based on trade type - blue for imports, orange for exports
              const opacity = Math.min(0.5 + partner.value / 10000, 0.95);
              
              // Use blue for imports and orange for exports
              let lineColor;
              if (partner.type === 'import') {
                // Blue color for imports with opacity
                lineColor = `rgba(59, 130, 246, ${opacity})`;  // Tailwind blue-500
              } else {
                // Orange color for exports with opacity
                lineColor = `rgba(249, 115, 22, ${opacity})`;  // Bright orange
              }
              
              return {
                startLat: srcCoords[0],
                startLng: srcCoords[1],
                endLat: targetCoords[0],
                endLng: targetCoords[1],
                color: lineColor,
                value: partner.value,
                tradePartner: partner.country,
                tradeValue: partner.value,
                tradeType: partner.type
              };
            })
            .filter(arc => arc !== null);
            
          setArcsData(newArcs);
          
          // Update the labels to show only selected country and partners
          if (showLabels) {
            const partnerCountries = newArcs.map(arc => arc.tradePartner);
            const filteredLabelData = generateLabelData([selectedCountry, ...partnerCountries]);
            setLabelData(filteredLabelData);
          }
        }, 100);
      })
      .catch(error => {
        console.error('Error fetching trade partners:', error);
      });
  };

  // Close the history panel and clear arcs
  const closeHistoryPanel = () => {
    setSelectedCountry(null);
    setCountryHistoryData([]);
    setArcsData([]); // Clear arc data when closing the panel
    setTopTradeRelationships([]); // Clear trade relationships when closing the panel
    setTradePartners([]); // Clear trade relationships when closing the panel
    
    // Restore all country labels when closing history panel
    if (showLabels) {
      setLabelData(generateLabelData());
    }
  };

  // Define color schemes - adding an orange-focused scheme for requested colors
  const colorSchemes = {
    orangeHeat: ["#ffedd5", "#fed7aa", "#fb923c", "#ea580c", "#9a3412", "#7c2d12"], // Light to dark orange
    viridis: ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"], // Purple to Yellow
    inferno: ["#000004", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"], // Black to Yellow
    plasma: ["#0d0887", "#5c01a6", "#9c179e", "#ed7953", "#fdb42f", "#f0f921"], // Indigo to Yellow
    magma: ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcffa4"], // Black to Light Yellow
    greens: ["#00441b", "#006d2c", "#238b45", "#41ab5d", "#78c679", "#c7e9c0"], // Dark Green to Light Green
    emerald: ["#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#ca6702"], // Teal to Orange
  };
  
  // Get color for dots based on ratio
  const getDotColor = (ratio) => {
    const colors = colorSchemes[colorScheme];
    
    // More granular color steps for better differentiation
    if (ratio < 0.15) {
      return colors[0];
    } else if (ratio < 0.30) {
      return colors[1];
    } else if (ratio < 0.50) {
      return colors[2];
    } else if (ratio < 0.75) {
      return colors[3];
    } else {
      return colors[4];
    }
  };

  // Get color for dots based on category (1-5, with 5 being highest)
  const getDotColorByCategory = (category) => {
    const colors = colorSchemes[colorScheme];
    
    // Map category 1-5 directly to colors array
    // We subtract 1 because categories are 1-5, but array indices are 0-4
    return colors[Math.min(category - 1, colors.length - 1)];
  };

  // NEW FUNCTION: Get color for country based on category (0-5, with 5 being highest)
  const getCountryColorByCategory = (category) => {
    const colors = colorSchemes[colorScheme];
    return colors[Math.min(category, colors.length - 1)];
  };

  // Format expenditure value based on data mode
  const formatExpenditureValue = (value, mode = dataMode) => {
    if (!value) return "No data";
    if (mode === 'gdp') {
      return `${value.toFixed(2)}% of GDP`;
    } else {
      return `$${value.toLocaleString()} million`;
    }
  };

  if (loading) return <div>Loading globe data...</div>;
  if (error) return <div>Error: {error}</div>;

  const yearExpenditures = expenditureData[selectedYear] || {};
  const yearGdpPercentages = gdpData[selectedYear] || {};

  return (
    <div
      className="globe-container"
      style={{ 
        height: "100vh", 
        width: "100vw", 
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "#000", // Keep black background
        overflow: "hidden", // Prevent scrollbars
        margin: 0,
        padding: 0,
        zIndex: 0 // Ensure the globe container is at the base layer
      }}
    >
      {geoData && (
        <Globe
          ref={globeRef}
          globeImageUrl="//unpkg.com/three-globe/example/img/earth-dark.jpg" // Using dark earth as base
          bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
          backgroundImageUrl={null}
          backgroundColor="#000000" // Black background
          lineHoverPrecision={0}
          
          // Custom appearance for countries using orange intensity color
          polygonsData={geoData.features}
          polygonCapColor={feature => {
            if (!feature.properties || !feature.properties.Country) return "#333333";
            const country = feature.properties.Country;
            return countryColors[country] || "#333333";
          }}
          polygonSideColor={() => "#222222"} // Slightly darker sides
          polygonStrokeColor={() => "#444444"} // Light grey borders
          polygonLabel={feature => {
            if (!feature.properties || !feature.properties.Country) return "";
            const country = feature.properties.Country;
            const value = dataMode === 'gdp' ? 
              yearGdpPercentages[country] : 
              yearExpenditures[country];
            
            if (!value) return `<div>${country}: No data</div>`;
            
            return `
              <div style="text-align:center;background:white;color:black;padding:8px;border-radius:4px;box-shadow:0 0 5px rgba(0,0,0,0.3);font-family:Arial,sans-serif;">
                <b>${country}</b><br />
                Year: ${selectedYear}<br />
                ${dataMode === 'gdp' ? '% of GDP' : 'Military Expenditure'}:<br />
                ${formatExpenditureValue(value)}
              </div>
            `;
          }}
          onPolygonClick={handleCountryClick}
          
          // Points for dots visualization (hidden in country-color mode)
          pointsData={visualizationMode === 'dots' ? pointsData : []}
          pointColor={d => d.color}
          pointAltitude={0.01}
          pointRadius={d => d.size}
          pointsMerge={false}
          pointLabel={d => `
            <div style="text-align:center;background:white;color:black;padding:8px;border-radius:4px;box-shadow:0 0 5px rgba(0,0,0,0.3);font-family:Arial,sans-serif;">
              <b>${d.country}</b><br />
              Year: ${selectedYear}<br />
              ${dataMode === 'gdp' ? '% of GDP' : 'Military Expenditure'}:<br />
              ${formatExpenditureValue(d.value)}
            </div>
          `}
          onPointClick={handleCountryClick}
          
          // Rings visualization
          ringsData={visualizationMode === 'rings' ? ringsData : []}
          ringColor={d => d.color}
          ringMaxRadius={d => d.radius * 1.5}
          ringPropagationSpeed={4}
          ringRepeatPeriod={800}
          ringsAltitude={0.01}
          onRingClick={handleCountryClick}
          
          // Pulses visualization
          pulsesData={visualizationMode === 'pulse' ? pulsesData : []}
          pulseAltitude={0.01}
          pulseColor={d => d.color}
          pulseRadial={true}
          pulseAnimDuration={1800}
          pulseRadius={0.7}
          onPulseClick={handleCountryClick}
          
          // Globe appearance
          atmosphereColor="#222222" // Dark atmosphere
          atmosphereAltitude={0.15}
          
          // Labels with dynamic filtering
          labelsData={labelData}
          labelLat={d => d.lat}
          labelLng={d => d.lng}
          labelText={d => d.text}
          labelSize={d => d.size}
          labelDotRadius={0.2} // Smaller dot
          labelColor={d => d.color || '#ffffff'} // Use dynamic color
          labelResolution={2}
          labelAltitude={0.01} // Keep close to surface
          labelIncludeDot={true} // Show the dot with the label
          labelLabel={d => `
            <div style="text-align:center;background:rgba(0,0,0,0.7);color:white;padding:3px 6px;border-radius:3px;font-family:Arial,sans-serif;font-size:10px;">
              ${d.text}
            </div>
          `}

          width={window.innerWidth}
          height={window.innerHeight}
          // Important: Make sure the Globe fills the entire viewport
          rendererConfig={{ 
            antialias: true, 
            alpha: true,
            preserveDrawingBuffer: false,
            precision: 'highp'
          }}

          // Arcs for trade flows - configure for one-time animation without repeating
          arcsData={arcsData}
          arcColor={d => d.color}
          arcDashLength={1}  // Use solid lines instead of dashed
          arcDashGap={0}     // No gap between dashes (solid line)
          arcDashAnimateTime={0} // No animation time (static display)
          arcStroke={d => 1.5 + (d.value / 8000)}  // Thicker lines for better visibility
          arcAltitude={0.35} // Higher arcs for better visibility
          arcCurveResolution={64} // Higher resolution curves
          arcCircularResolution={32} // Higher resolution circles
          arcLabel={d => `
            <div style="text-align:center;background:white;color:black;padding:8px;border-radius:4px;box-shadow:0 0 5px rgba(0,0,0,0.3);font-family:Arial,sans-serif;">
              <b>${d.tradePartner}</b><br />
              ${d.tradeType.charAt(0).toUpperCase() + d.tradeType.slice(1)} value: $${d.tradeValue.toLocaleString()} million
            </div>
          `}
        />
      )}

      {/* Year slider control - updated to dark theme with orange accents */}
      <div
        style={{
          position: "absolute",
          bottom: "20px",
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(17, 17, 17, 0.8)",
          padding: "15px 20px",
          borderRadius: "8px",
          boxShadow: "0 0 15px rgba(0, 0, 0, 0.5)",
          border: "1px solid #333333",
          zIndex: 1000,
          width: "80%",
          maxWidth: "500px",
          backdropFilter: "blur(5px)"
        }}
      >
        <div style={{ marginBottom: "8px", textAlign: "center", color: "#ea580c", fontFamily: "Arial, sans-serif", fontWeight: "bold" }}>
          Year: {selectedYear}
        </div>
        <input
          type="range"
          min="1988"
          max="2022"
          value={selectedYear}
          onChange={(e) => {
            const newYear = parseInt(e.target.value);
            setSelectedYear(newYear);
            console.log(`Year changed to ${newYear}`);
          }}
          style={{ 
            width: "100%",
            accentColor: "#ea580c", // Orange accent color
            height: "8px",
            background: "#333333",
            borderRadius: "4px",
            outline: "none",
            opacity: "0.9",
            transition: "opacity 0.2s"
          }}
        />
      </div>

      {/* Legend for country colors */}
      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          background: "rgba(17, 17, 17, 0.8)",
          padding: "15px",
          borderRadius: "8px",
          boxShadow: "0 0 15px rgba(0, 0, 0, 0.5)",
          border: "1px solid #333333",
          zIndex: 1000,
          color: "#ffffff",
          fontFamily: "Arial, sans-serif",
          backdropFilter: "blur(5px)",
          display: selectedCountry ? "none" : "block"
        }}
      >
        <h3 style={{ margin: "0 0 15px 0", textAlign: "center", color: "#ea580c" }}>
          Military Expenditure
        </h3>
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {[5, 4, 3, 2, 1, 0].map(category => (
            <div 
              key={`legend-${category}`}
              style={{ 
                display: "flex", 
                alignItems: "center",
                gap: "10px"
              }}
            >
              <div
                style={{
                  width: "20px",
                  height: "20px",
                  backgroundColor: getCountryColorByCategory(category),
                  borderRadius: "3px"
                }}
              ></div>
              <span>
                {category === 5 ? "Top 5%" : 
                 category === 4 ? "90-95 percentile" :
                 category === 3 ? "75-90 percentile" :
                 category === 2 ? "50-75 percentile" :
                 category === 1 ? "25-50 percentile" :
                 "Bottom 25%"}
              </span>
            </div>
          ))}
          <div 
            style={{ 
              display: "flex", 
              alignItems: "center",
              gap: "10px",
              marginTop: "5px"
            }}
          >
            <div
              style={{
                width: "20px",
                height: "20px",
                backgroundColor: "#333333",
                borderRadius: "3px"
              }}
            ></div>
            <span>No data</span>
          </div>
        </div>
      </div>

      {/* Legend for country colors - always visible */}
      <div
        style={{
          position: "absolute",
          bottom: "0px",
          left: "20px",
          background: "rgba(17, 17, 17, 0.8)",
          padding: "15px",
          borderRadius: "8px",
          boxShadow: "0 0 15px rgba(0, 0, 0, 0.5)",
          border: "1px solid #333333",
          zIndex: 1000,
          color: "#ffffff",
          fontFamily: "Arial, sans-serif",
          backdropFilter: "blur(5px)"
        }}
      >
        <h3 style={{ margin: "0 0 15px 0", textAlign: "center", color: "#ea580c" }}>
          Military Expenditure
        </h3>
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {[5, 4, 3, 2, 1, 0].map(category => (
            <div 
              key={`legend-${category}`}
              style={{ 
                display: "flex", 
                alignItems: "center",
                gap: "10px"
              }}
            >
              <div
                style={{
                  width: "20px",
                  height: "20px",
                  backgroundColor: getCountryColorByCategory(category),
                  borderRadius: "3px"
                }}
              ></div>
              <span>
                {category === 5 ? "Top 5%" : 
                 category === 4 ? "90-95 percentile" :
                 category === 3 ? "75-90 percentile" :
                 category === 2 ? "50-75 percentile" :
                 category === 1 ? "25-50 percentile" :
                 "Bottom 25%"}
              </span>
            </div>
          ))}
          <div 
            style={{ 
              display: "flex", 
              alignItems: "center",
              gap: "10px",
              marginTop: "5px"
            }}
          >
            <div
              style={{
                width: "20px",
                height: "20px",
                backgroundColor: "#333333",
                borderRadius: "3px"
              }}
            ></div>
            <span>No data</span>
          </div>
        </div>
      </div>

      {/* Top countries panel - updated to dark theme with orange accents */}
      <div
        style={{
          position: "absolute",
          top: "70px",
          left: "20px",
          background: "rgba(17, 17, 17, 0.8)",
          padding: "15px",
          borderRadius: "8px",
          boxShadow: "0 0 15px rgba(0, 0, 0, 0.5)",
          border: "1px solid #333333",
          zIndex: 1000,
          width: "300px",
          display: selectedCountry ? "none" : "block",
          color: "#ffffff",
          fontFamily: "Arial, sans-serif",
          backdropFilter: "blur(5px)"
        }}
      >
        <h3 style={{ margin: "0 0 15px 0", textAlign: "center", color: "#ea580c" }}>Top Military Spenders</h3>
        <div style={{ maxHeight: "400px", overflowY: "auto" }}>
          {topCountries.map((item, index) => (
            <div 
              key={index}
              style={{ 
                display: "flex", 
                justifyContent: "space-between", 
                marginBottom: "8px",
                padding: "8px 12px",
                borderRadius: "4px",
                backgroundColor: index % 2 === 0 ? "rgba(51, 51, 51, 0.5)" : "rgba(34, 34, 34, 0.5)",
                border: "1px solid #444444"
              }}
            >
              <div style={{ color: "#cccccc" }}>{index + 1}. {item.country}</div>
              <div style={{ fontWeight: "bold", color: "#ea580c" }}>
                {formatExpenditureValue(item.value, 'total')}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* New Top 5 Trade Relationships Panel */}
      {selectedCountry && topTradeRelationships.length > 0 && (
        <div
          style={{
            position: "absolute",
            top: "70px",
            right: "20px",
            background: "rgba(17, 17, 17, 0.8)",
            padding: "15px",
            borderRadius: "8px",
            boxShadow: "0 0 15px rgba(0, 0, 0, 0.5)",
            border: "1px solid #333333",
            zIndex: 1000,
            width: "300px",
            color: "#ffffff",
            fontFamily: "Arial, sans-serif",
            backdropFilter: "blur(5px)"
          }}
        >
          <h3 style={{ margin: "0 0 15px 0", textAlign: "center", color: "#ea580c" }}>
            Top 5 Trade Relationships
          </h3>
          <div style={{ maxHeight: "400px", overflowY: "auto" }}>
            {topTradeRelationships.map((item, index) => (
              <div 
                key={index}
                style={{ 
                  display: "flex", 
                  flexDirection: "column",
                  marginBottom: "12px",
                  padding: "10px",
                  borderRadius: "4px",
                  backgroundColor: index % 2 === 0 ? "rgba(51, 51, 51, 0.5)" : "rgba(34, 34, 34, 0.5)",
                  border: "1px solid #444444"
                }}
              >
                <div style={{ 
                  display: "flex", 
                  justifyContent: "space-between",
                  marginBottom: "5px",
                  borderBottom: "1px solid #444",
                  paddingBottom: "5px"
                }}>
                  <div style={{ color: "#ffffff", fontWeight: "bold" }}>{index + 1}. {item.country}</div>
                  <div style={{ fontWeight: "bold", color: "#ea580c" }}>
                    ${item.value.toLocaleString()} million
                  </div>
                </div>
                <div style={{ 
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "0.9em"
                }}>
                  <div style={{ color: "#cccccc" }}>Type:</div>
                  <div style={{ color: "#ffffff", textTransform: "capitalize" }}>{item.type}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Trade Partners Panel - Shows top 5 arms trade relationships */}
      {selectedCountry && tradePartners.length > 0 && (
        <div
          style={{
            position: "absolute",
            top: "70px",
            right: "20px",
            background: "rgba(17, 17, 17, 0.8)",
            padding: "15px",
            borderRadius: "8px",
            boxShadow: "0 0 15px rgba(0, 0, 0, 0.5)",
            border: "1px solid #333333",
            zIndex: 1000,
            width: "360px",
            color: "#ffffff",
            fontFamily: "Arial, sans-serif",
            backdropFilter: "blur(5px)"
          }}
        >
          <h3 style={{ margin: "0 0 15px 0", textAlign: "center", color: "#ea580c" }}>
            Top 5 Arms Trade Partners
          </h3>
          <div style={{ maxHeight: "400px", overflowY: "auto" }}>
            {tradePartners.map((partner, index) => (
              <div 
                key={index}
                style={{ 
                  display: "flex", 
                  flexDirection: "column",
                  marginBottom: "15px",
                  padding: "12px",
                  borderRadius: "4px",
                  backgroundColor: index % 2 === 0 ? "rgba(51, 51, 51, 0.5)" : "rgba(34, 34, 34, 0.5)",
                  border: "1px solid #444444"
                }}
              >
                <div style={{ 
                  display: "flex", 
                  justifyContent: "space-between",
                  marginBottom: "10px",
                  borderBottom: "1px solid #444",
                  paddingBottom: "8px"
                }}>
                  <div style={{ color: "#ffffff", fontWeight: "bold", fontSize: "1.1em" }}>
                    {index + 1}. {partner.country}
                  </div>
                  <div style={{ fontWeight: "bold", color: "#ea580c" }}>
                    ${partner.total.toLocaleString()} million
                  </div>
                </div>
                <div style={{ 
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: "5px",
                  padding: "5px 0"
                }}>
                  <div style={{ color: "#cccccc" }}>Imports from {partner.country}:</div>
                  <div style={{ color: "#ffffff", fontWeight: "bold" }}>
                    ${partner.imports.toLocaleString()} million
                  </div>
                </div>
                <div style={{ 
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "5px 0"
                }}>
                  <div style={{ color: "#cccccc" }}>Exports to {partner.country}:</div>
                  <div style={{ color: "#ffffff", fontWeight: "bold" }}>
                    ${partner.exports.toLocaleString()} million
                  </div>
                </div>
                {/* Visual bar showing the import/export ratio */}
                <div style={{ 
                  display: "flex", 
                  height: "8px", 
                  marginTop: "10px",
                  backgroundColor: "#222", 
                  borderRadius: "4px", 
                  overflow: "hidden" 
                }}>
                  <div 
                    style={{ 
                      width: `${partner.total > 0 ? (partner.imports / partner.total) * 100 : 0}%`,
                      backgroundColor: "#3b82f6", // blue for imports
                      height: "100%" 
                    }} 
                  />
                  <div 
                    style={{ 
                      width: `${partner.total > 0 ? (partner.exports / partner.total) * 100 : 0}%`,
                      backgroundColor: "#ea580c", // orange for exports
                      height: "100%" 
                    }} 
                  />
                </div>
                <div style={{ 
                  display: "flex", 
                  justifyContent: "space-between", 
                  fontSize: "0.8em",
                  marginTop: "3px"
                }}>
                  <span style={{ color: "#3b82f6" }}>Imports</span>
                  <span style={{ color: "#ea580c" }}>Exports</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Country history panel - updated to dark theme with orange accents */}
      {selectedCountry && (
        <div
          style={{
            position: "absolute",
            top: "70px",
            left: "20px",
            background: "rgba(17, 17, 17, 0.8)",
            padding: "15px",
            borderRadius: "8px",
            boxShadow: "0 0 15px rgba(0, 0, 0, 0.5)",
            border: "1px solid #333333",
            zIndex: 1000,
            width: "450px",
            maxHeight: "80vh",
            overflow: "auto",
            color: "#ffffff",
            fontFamily: "Arial, sans-serif",
            backdropFilter: "blur(5px)"
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
            <h3 style={{ margin: 0, color: "#ea580c" }}>{selectedCountry} - Historical Data</h3>
            <button
              onClick={closeHistoryPanel}
              style={{
                background: "rgba(51, 51, 51, 0.5)",
                border: "1px solid #444444",
                color: "#cccccc",
                width: "25px",
                height: "25px",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "16px",
                cursor: "pointer",
              }}
            >
              ×
            </button>
          </div>
          
          {/* Show both total and GDP data side by side */}
          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            <div>
              <h4 style={{ marginTop: 0, color: "#cccccc" }}>
                {dataMode === 'gdp' ? '% of GDP' : 'Total Expenditure'}
              </h4>
              <ChartComponent
                timeSeriesData={countryHistoryData}
                selectedCountry={selectedCountry}
                dataMode={dataMode}
              />
            </div>
            
            <div>
              <button
                onClick={() => window.dispatchEvent(new CustomEvent('setDataMode', { 
                  detail: dataMode === 'gdp' ? 'total' : 'gdp' 
                }))}
                style={{
                  padding: "10px",
                  backgroundColor: "#ea580c",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                  width: "100%",
                }}
              >
                Switch to {dataMode === 'gdp' ? 'Total Expenditure' : '% of GDP'} View
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Globe controls - updated to dark theme with orange accents */}
      <div
        style={{
          position: "absolute",
          bottom: "100px",
          right: "20px",
          background: "rgba(17, 17, 17, 0.8)",
          padding: "10px",
          borderRadius: "8px",
          boxShadow: "0 0 15px rgba(0, 0, 0, 0.5)",
          border: "1px solid #333333",
          zIndex: 1000,
          backdropFilter: "blur(5px)"
        }}
      >
        <div style={{ display: "flex" }}>
          <button 
            onClick={() => setAltitude(prev => Math.min(prev + 0.5, 4))}
            style={{
              padding: "5px 15px",
              margin: "0 5px",
              cursor: "pointer",
              backgroundColor: "#333333",
              color: "white",
              border: "1px solid #444444",
              borderRadius: "4px",
            }}
          >
            Zoom Out
          </button>
          <button 
            onClick={() => setAltitude(prev => Math.max(prev - 0.5, 1))}
            style={{
              padding: "5px 15px",
              margin: "0 5px",
              cursor: "pointer",
              backgroundColor: "#333333",
              color: "white",
              border: "1px solid #444444",
              borderRadius: "4px",
            }}
          >
            Zoom In
          </button>
        </div>
      </div>

      {/* Trade view toggle button - only visible when a country is selected */}
      {selectedCountry && (
        <div
          style={{
            position: "absolute",
            bottom: "20px",
            right: "20px",
            background: "rgba(17, 17, 17, 0.85)",
            padding: "10px 15px",
            borderRadius: "8px",
            boxShadow: "0 0 15px rgba(0, 0, 0, 0.5)",
            border: "1px solid #333333",
            zIndex: 1000,
            backdropFilter: "blur(5px)",
            fontFamily: "Arial, sans-serif",
          }}
        >
          <div style={{ marginBottom: "8px", color: "#cccccc", fontSize: "14px", textAlign: "center" }}>
            Trade View
          </div>
          <div style={{ 
            display: "flex", 
            borderRadius: "4px", 
            overflow: "hidden", 
            border: "1px solid #444444" 
          }}>
            <button
              onClick={() => setTradeViewMode('all')}
              style={{
                padding: "6px 12px",
                backgroundColor: tradeViewMode === 'all' ? "#ea580c" : "#333333",
                color: "white",
                border: "none",
                borderRight: "1px solid #444444",
                cursor: "pointer",
                fontSize: "13px",
                flex: 1,
                whiteSpace: "nowrap",
              }}
            >
              All
            </button>
            <button
              onClick={() => setTradeViewMode('import')}
              style={{
                padding: "6px 12px",
                backgroundColor: tradeViewMode === 'import' ? "#3b82f6" : "#333333",
                color: "white",
                border: "none",
                borderRight: "1px solid #444444",
                cursor: "pointer",
                fontSize: "13px",
                flex: 1,
                whiteSpace: "nowrap",
              }}
            >
              Imports
            </button>
            <button
              onClick={() => setTradeViewMode('export')}
              style={{
                padding: "6px 12px",
                backgroundColor: tradeViewMode === 'export' ? "#ea580c" : "#333333",
                color: "white",
                border: "none",
                cursor: "pointer",
                fontSize: "13px",
                flex: 1,
                whiteSpace: "nowrap",
              }}
            >
              Exports
            </button>
          </div>
        </div>
      )}
      {/* Legend for country colors - always visible */}
      <div
        style={{
          position: "absolute",
          bottom: "0px",
          left: "20px",
          background: "rgba(17, 17, 17, 0.8)",
          padding: "15px",
          borderRadius: "8px",
          boxShadow: "0 0 15px rgba(0, 0, 0, 0.5)",
          border: "1px solid #333333",
          zIndex: 1000,
          color: "#ffffff",
          fontFamily: "Arial, sans-serif",
          backdropFilter: "blur(5px)"
        }}
      >
        <h3 style={{ margin: "0 0 15px 0", textAlign: "center", color: "#ea580c" }}>
          Military Expenditure
        </h3>
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {[5, 4, 3, 2, 1, 0].map(category => (
            <div 
              key={`legend-${category}`}
              style={{ 
                display: "flex", 
                alignItems: "center",
                gap: "10px"
              }}
            >
              <div
                style={{
                  width: "20px",
                  height: "20px",
                  backgroundColor: getCountryColorByCategory(category),
                  borderRadius: "3px"
                }}
              ></div>
              <span>
                {category === 5 ? "Top 5%" : 
                 category === 4 ? "90-95 percentile" :
                 category === 3 ? "75-90 percentile" :
                 category === 2 ? "50-75 percentile" :
                 category === 1 ? "25-50 percentile" :
                 "Bottom 25%"}
              </span>
            </div>
          ))}
          <div 
            style={{ 
              display: "flex", 
              alignItems: "center",
              gap: "10px",
              marginTop: "5px"
            }}
          >
            <div
              style={{
                width: "20px",
                height: "20px",
                backgroundColor: "#333333",
                borderRadius: "3px"
              }}
            ></div>
            <span>No data</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GlobeComponent;