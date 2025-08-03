import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

const MapperVisualization = ({ mapperData, width = 600, height = 400 }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!mapperData || !mapperData.nodes || !mapperData.links) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous content

    // Create main group
    const g = svg.append("g");

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
      });

    svg.call(zoom);

    // Create force simulation
    const simulation = d3.forceSimulation(mapperData.nodes)
      .force("link", d3.forceLink(mapperData.links).id(d => d.id).distance(50))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(d => d.radius + 2));

    // Create links
    const link = g.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(mapperData.links)
      .enter().append("line")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", d => Math.sqrt(d.weight || 1));

    // Create nodes
    const node = g.append("g")
      .attr("class", "nodes")
      .selectAll("circle")
      .data(mapperData.nodes)
      .enter().append("circle")
      .attr("r", d => d.radius || 8)
      .attr("fill", d => d.color || "#69b3a2")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Add node labels
    const labels = g.append("g")
      .attr("class", "labels")
      .selectAll("text")
      .data(mapperData.nodes)
      .enter().append("text")
      .text(d => d.label || d.id)
      .attr("font-size", "10px")
      .attr("dx", 12)
      .attr("dy", 4);

    // Add tooltips
    node.append("title")
      .text(d => `Node: ${d.id}\nSize: ${d.size || 'N/A'}\nPoints: ${d.points?.length || 0}`);

    // Update positions on simulation tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      labels
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    });

    // Drag functions
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    return () => {
      simulation.stop();
    };

  }, [mapperData, width, height]);

  return (
    <div className="mapper-visualization">
      <h3>Mapper Network</h3>
      <p>Interactive network showing topological structure. Drag nodes to explore!</p>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{ border: '1px solid #ccc', background: '#f9f9f9' }}
      />
      <div className="mapper-controls" style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
        <p>• Zoom: Mouse wheel | • Pan: Click and drag background | • Move nodes: Drag circles</p>
        <p>• Node size represents cluster density | • Edge thickness shows connection strength</p>
      </div>
    </div>
  );
};

export default MapperVisualization;
