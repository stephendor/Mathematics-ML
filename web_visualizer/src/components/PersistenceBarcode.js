import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

const PersistenceBarcode = ({ persistenceData, maxFiltration = 1.0 }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!persistenceData || !persistenceData.pairs) return;

    const svg = d3.select(svgRef.current);
    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    // Clear existing content
    svg.selectAll("*").remove();

    // Create main group
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Process data - separate by dimension and sort by persistence
    const dimensions = [0, 1, 2];
    const colorScale = d3.scaleOrdinal()
      .domain(dimensions)
      .range(["#2563eb", "#dc2626", "#16a34a"]);

    let allBars = [];
    let yOffset = 0;
    const barHeight = 3;
    const barSpacing = 1;
    const dimensionSpacing = 20;

    dimensions.forEach(dim => {
      const dimData = persistenceData.pairs
        .filter(d => d.dimension === dim)
        .sort((a, b) => (b.death - b.birth) - (a.death - a.birth)); // Sort by persistence

      dimData.forEach((d, i) => {
        allBars.push({
          ...d,
          y: yOffset + i * (barHeight + barSpacing),
          color: colorScale(dim)
        });
      });

      yOffset += dimData.length * (barHeight + barSpacing) + dimensionSpacing;
    });

    // Set up scales
    const maxValue = Math.min(maxFiltration, d3.max(persistenceData.pairs, d => 
      d.death === Infinity ? maxFiltration : d.death
    ));

    const xScale = d3.scaleLinear()
      .domain([0, maxValue])
      .range([0, width]);

    // Draw bars
    g.selectAll(".persistence-bar")
      .data(allBars)
      .enter()
      .append("rect")
      .attr("class", "persistence-bar")
      .attr("x", d => xScale(d.birth))
      .attr("y", d => d.y)
      .attr("width", d => {
        const death = d.death === Infinity ? maxValue : d.death;
        return xScale(death) - xScale(d.birth);
      })
      .attr("height", barHeight)
      .attr("fill", d => d.color)
      .attr("opacity", 0.8)
      .on("mouseover", function(event, d) {
        d3.select(this).attr("opacity", 1);
        
        const tooltip = d3.select("body").append("div")
          .attr("class", "tooltip")
          .style("position", "absolute")
          .style("background", "rgba(0,0,0,0.8)")
          .style("color", "white")
          .style("padding", "8px")
          .style("border-radius", "4px")
          .style("font-size", "12px")
          .style("pointer-events", "none")
          .style("z-index", "1000");

        tooltip.html(`
          <div>Dimension: ${d.dimension}</div>
          <div>Birth: ${d.birth.toFixed(3)}</div>
          <div>Death: ${d.death === Infinity ? '∞' : d.death.toFixed(3)}</div>
          <div>Persistence: ${d.death === Infinity ? '∞' : (d.death - d.birth).toFixed(3)}</div>
        `);

        const [mouseX, mouseY] = d3.pointer(event, document.body);
        tooltip
          .style("left", (mouseX + 10) + "px")
          .style("top", (mouseY - 10) + "px");
      })
      .on("mouseout", function() {
        d3.select(this).attr("opacity", 0.8);
        d3.selectAll(".tooltip").remove();
      });

    // Add dimension labels
    yOffset = 0;
    dimensions.forEach(dim => {
      const dimData = persistenceData.pairs.filter(d => d.dimension === dim);
      
      if (dimData.length > 0) {
        g.append("text")
          .attr("x", -10)
          .attr("y", yOffset + (dimData.length * (barHeight + barSpacing)) / 2)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("font-size", "12px")
          .attr("font-weight", "bold")
          .attr("fill", colorScale(dim))
          .text(`H${dim}`);
        
        yOffset += dimData.length * (barHeight + barSpacing) + dimensionSpacing;
      }
    });

    // Add x-axis
    const xAxis = d3.axisBottom(xScale);
    
    g.append("g")
      .attr("transform", `translate(0, ${Math.max(yOffset - dimensionSpacing, 0)})`)
      .call(xAxis)
      .append("text")
      .attr("x", width / 2)
      .attr("y", 35)
      .attr("fill", "black")
      .style("text-anchor", "middle")
      .text("Filtration Value");

    // Add title
    svg.append("text")
      .attr("x", (width + margin.left + margin.right) / 2)
      .attr("y", 15)
      .attr("text-anchor", "middle")
      .attr("font-size", "16px")
      .attr("font-weight", "bold")
      .text("Persistence Barcode");

  }, [persistenceData, maxFiltration]);

  return (
    <div className="persistence-barcode">
      <svg 
        ref={svgRef} 
        width={600} 
        height={300}
        style={{ border: '1px solid #ddd', borderRadius: '4px' }}
      />
    </div>
  );
};

export default PersistenceBarcode;
