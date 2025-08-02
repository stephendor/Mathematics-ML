import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

const PersistenceDiagram = ({ persistenceData, maxFiltration = 1.0 }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!persistenceData || !persistenceData.pairs) return;

    const svg = d3.select(svgRef.current);
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const width = 400 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Clear existing content
    svg.selectAll("*").remove();

    // Create main group
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Set up scales
    const maxValue = Math.min(maxFiltration, d3.max(persistenceData.pairs, d => 
      d.death === Infinity ? maxFiltration : d.death
    ));

    const xScale = d3.scaleLinear()
      .domain([0, maxValue])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, maxValue])
      .range([height, 0]);

    // Add diagonal line (y = x)
    g.append("line")
      .attr("x1", xScale(0))
      .attr("y1", yScale(0))
      .attr("x2", xScale(maxValue))
      .attr("y2", yScale(maxValue))
      .attr("stroke", "#ef4444")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5")
      .attr("opacity", 0.7);

    // Color scale for dimensions
    const colorScale = d3.scaleOrdinal()
      .domain([0, 1, 2])
      .range(["#2563eb", "#dc2626", "#16a34a"]);

    // Filter out infinite points for initial display
    const finitePoints = persistenceData.pairs.filter(d => d.death !== Infinity);
    const infinitePoints = persistenceData.pairs.filter(d => d.death === Infinity);

    // Add finite persistence points
    g.selectAll(".persistence-point")
      .data(finitePoints)
      .enter()
      .append("circle")
      .attr("class", "persistence-point")
      .attr("cx", d => xScale(d.birth))
      .attr("cy", d => yScale(d.death))
      .attr("r", 4)
      .attr("fill", d => colorScale(d.dimension))
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .attr("opacity", 0.8)
      .on("mouseover", function(event, d) {
        // Add tooltip
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
          <div>Death: ${d.death.toFixed(3)}</div>
          <div>Persistence: ${(d.death - d.birth).toFixed(3)}</div>
        `);

        const [mouseX, mouseY] = d3.pointer(event, document.body);
        tooltip
          .style("left", (mouseX + 10) + "px")
          .style("top", (mouseY - 10) + "px");
      })
      .on("mouseout", function() {
        d3.selectAll(".tooltip").remove();
      });

    // Add infinite persistence points (shown on top edge)
    g.selectAll(".infinite-point")
      .data(infinitePoints)
      .enter()
      .append("circle")
      .attr("class", "infinite-point")
      .attr("cx", d => xScale(d.birth))
      .attr("cy", 5)
      .attr("r", 6)
      .attr("fill", d => colorScale(d.dimension))
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .on("mouseover", function(event, d) {
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
          <div>Death: ∞</div>
          <div>Persistence: ∞</div>
        `);

        const [mouseX, mouseY] = d3.pointer(event, document.body);
        tooltip
          .style("left", (mouseX + 10) + "px")
          .style("top", (mouseY - 10) + "px");
      })
      .on("mouseout", function() {
        d3.selectAll(".tooltip").remove();
      });

    // Add axes
    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale);

    g.append("g")
      .attr("transform", `translate(0, ${height})`)
      .call(xAxis)
      .append("text")
      .attr("x", width / 2)
      .attr("y", 35)
      .attr("fill", "black")
      .style("text-anchor", "middle")
      .text("Birth");

    g.append("g")
      .call(yAxis)
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -30)
      .attr("x", -height / 2)
      .attr("fill", "black")
      .style("text-anchor", "middle")
      .text("Death");

    // Add title
    svg.append("text")
      .attr("x", (width + margin.left + margin.right) / 2)
      .attr("y", 15)
      .attr("text-anchor", "middle")
      .attr("font-size", "16px")
      .attr("font-weight", "bold")
      .text("Persistence Diagram");

    // Add legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width + margin.left - 100}, 40)`);

    const dimensions = [0, 1, 2];
    const dimensionNames = ["Connected Components", "Loops", "Voids"];

    legend.selectAll(".legend-item")
      .data(dimensions)
      .enter()
      .append("g")
      .attr("class", "legend-item")
      .attr("transform", (d, i) => `translate(0, ${i * 20})`)
      .each(function(d, i) {
        const item = d3.select(this);
        
        item.append("circle")
          .attr("cx", 6)
          .attr("cy", 0)
          .attr("r", 4)
          .attr("fill", colorScale(d));
        
        item.append("text")
          .attr("x", 15)
          .attr("y", 4)
          .attr("font-size", "12px")
          .text(`H${d}: ${dimensionNames[i]}`);
      });

  }, [persistenceData, maxFiltration]);

  return (
    <div className="persistence-diagram">
      <svg 
        ref={svgRef} 
        width={400} 
        height={400}
        style={{ border: '1px solid #ddd', borderRadius: '4px' }}
      />
    </div>
  );
};

export default PersistenceDiagram;
