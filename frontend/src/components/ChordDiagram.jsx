// frontend/src/components/ChordDiagram.jsx
import { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';

function ChordDiagram({ data, selectedYear }) {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  const [dimensions, setDimensions] = useState({
    width: 800,
    height: 800
  });

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      const containerWidth = svgRef.current.parentElement.clientWidth;
      const size = Math.min(containerWidth, window.innerHeight * 0.8);
      setDimensions({
        width: size,
        height: size
      });
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Initial sizing

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Create and update the chord diagram
  useEffect(() => {
    if (!data || !svgRef.current) return;

    const { countries, matrix } = data;
    const { width, height } = dimensions;

    // Clear previous SVG content
    d3.select(svgRef.current).selectAll('*').remove();

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`);

    // Create tooltip
    const tooltip = d3.select(tooltipRef.current)
      .style('opacity', 0)
      .attr('class', 'tooltip')
      .style('background-color', 'white')
      .style('border', 'solid')
      .style('border-width', '2px')
      .style('border-radius', '5px')
      .style('padding', '5px');

    // Create color scale
    const color = d3.scaleOrdinal(d3.schemeCategory10);

    // Create chord layout
    const radius = Math.min(width, height) * 0.4;
    const chord = d3.chord()
      .padAngle(0.05)
      .sortSubgroups(d3.descending);

    const chords = chord(matrix);

    // Create arc generator
    const arc = d3.arc()
      .innerRadius(radius)
      .outerRadius(radius + 20);

    // Create ribbon generator
    const ribbon = d3.ribbon()
      .radius(radius);

    // Add groups (countries)
    const group = svg.append('g')
      .selectAll('g')
      .data(chords.groups)
      .join('g');

    // Add arcs for each country
    group.append('path')
      .attr('d', arc)
      .style('fill', d => color(d.index))
      .style('stroke', 'white')
      .style('opacity', 0.8)
      .on('mouseover', function(event, d) {
        // Highlight this country's connections
        d3.select(this).style('opacity', 1);
        
        // Highlight related ribbons
        svg.selectAll('.ribbon')
          .style('opacity', ribbon => {
            return (ribbon.source.index === d.index || ribbon.target.index === d.index) ? 0.9 : 0.1;
          });
        
        // Show tooltip
        tooltip
          .style('opacity', 1)
          .html(`<strong>${countries[d.index]}</strong>`)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseout', function() {
        // Reset opacity
        d3.select(this).style('opacity', 0.8);
        svg.selectAll('.ribbon').style('opacity', 0.6);
        
        // Hide tooltip
        tooltip.style('opacity', 0);
      });

    // Add country labels
    group.append('text')
      .each(d => { d.angle = (d.startAngle + d.endAngle) / 2; })
      .attr('dy', '0.35em')
      .attr('transform', d => `
        rotate(${(d.angle * 180 / Math.PI - 90)})
        translate(${radius + 30})
        ${d.angle > Math.PI ? 'rotate(180)' : ''}
      `)
      .attr('text-anchor', d => d.angle > Math.PI ? 'end' : 'start')
      .text(d => countries[d.index])
      .style('font-size', '12px')
      .style('font-weight', 'bold');

    // Add ribbons (trade flows)
    svg.append('g')
      .selectAll('path')
      .data(chords)
      .join('path')
      .attr('class', 'ribbon')
      .attr('d', ribbon)
      .style('fill', d => color(d.source.index))
      .style('stroke', 'white')
      .style('stroke-width', 0.1)
      .style('opacity', 0.6)
      .on('mouseover', function(event, d) {
        // Highlight this ribbon
        d3.select(this).style('opacity', 1);
        
        // Show tooltip with trade details
        const sourceCountry = countries[d.source.index];
        const targetCountry = countries[d.target.index];
        const value = d.source.value;
        
        tooltip
          .style('opacity', 1)
          .html(`
            <strong>From:</strong> ${sourceCountry}<br>
            <strong>To:</strong> ${targetCountry}<br>
            <strong>Value:</strong> ${value.toLocaleString()} SIPRI TIV
          `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseout', function() {
        // Reset opacity
        d3.select(this).style('opacity', 0.6);
        
        // Hide tooltip
        tooltip.style('opacity', 0);
      });

    // Add title
    svg.append('text')
      .attr('x', 0)
      .attr('y', -height / 2 + 20)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text(`Arms Trade Flows (${selectedYear})`);

  }, [data, dimensions, selectedYear]);

  return (
    <div className="chord-diagram-container">
      <svg ref={svgRef}></svg>
      <div ref={tooltipRef} className="tooltip"></div>
    </div>
  );
}

export default ChordDiagram;