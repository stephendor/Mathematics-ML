import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';

const PointCloudCanvas = ({ points, onPointsChange, filtrationValue = 0.5 }) => {
  const svgRef = useRef();
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const width = 600;
    const height = 400;

    // Clear existing content
    svg.selectAll("*").remove();

    // Set up scales
    const xScale = d3.scaleLinear().domain([0, 1]).range([50, width - 50]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([height - 50, 50]);

    // Add background rectangle for click handling
    svg.append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "white")
      .attr("stroke", "#ddd")
      .on("click", (event) => {
        if (!isDrawing) return;
        
        const [mouseX, mouseY] = d3.pointer(event);
        const newPoint = {
          x: xScale.invert(mouseX),
          y: yScale.invert(mouseY),
          id: Date.now()
        };
        
        onPointsChange([...points, newPoint]);
      });

    // Draw points
    svg.selectAll(".point")
      .data(points)
      .enter()
      .append("circle")
      .attr("class", "point")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", 4)
      .attr("fill", "#2563eb")
      .attr("stroke", "#1e40af")
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        if (event.shiftKey) {
          // Remove point on shift+click
          onPointsChange(points.filter(p => p.id !== d.id));
        }
      });

    // Draw edges based on filtration value
    const edges = [];
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const dist = Math.sqrt(
          Math.pow(points[i].x - points[j].x, 2) + 
          Math.pow(points[i].y - points[j].y, 2)
        );
        if (dist <= filtrationValue) {
          edges.push({
            source: points[i],
            target: points[j],
            distance: dist
          });
        }
      }
    }

    svg.selectAll(".edge")
      .data(edges)
      .enter()
      .append("line")
      .attr("class", "edge")
      .attr("x1", d => xScale(d.source.x))
      .attr("y1", d => yScale(d.source.y))
      .attr("x2", d => xScale(d.target.x))
      .attr("y2", d => yScale(d.target.y))
      .attr("stroke", "#94a3b8")
      .attr("stroke-width", 1)
      .attr("opacity", 0.6);

    // Add axes
    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale);

    svg.append("g")
      .attr("transform", `translate(0, ${height - 50})`)
      .call(xAxis);

    svg.append("g")
      .attr("transform", `translate(50, 0)`)
      .call(yAxis);

  }, [points, filtrationValue, isDrawing, onPointsChange]);

  return (
    <div className="point-cloud-container">
      <div className="controls">
        <button 
          onClick={() => setIsDrawing(!isDrawing)}
          className={`btn ${isDrawing ? 'btn-active' : 'btn-inactive'}`}
        >
          {isDrawing ? 'Stop Drawing' : 'Draw Points'}
        </button>
        <button 
          onClick={() => onPointsChange([])}
          className="btn btn-danger"
        >
          Clear All
        </button>
        <span className="instructions">
          {isDrawing ? 'Click to add points, Shift+Click to remove' : 'Enable drawing mode to add points'}
        </span>
      </div>
      <svg 
        ref={svgRef} 
        width={600} 
        height={400}
        style={{ border: '1px solid #ddd', borderRadius: '4px' }}
      />
    </div>
  );
};

export default PointCloudCanvas;
