import React, { useState, useEffect, useCallback } from 'react';
import PointCloudCanvas from './components/PointCloudCanvas';
import PersistenceDiagram from './components/PersistenceDiagram';
import PersistenceBarcode from './components/PersistenceBarcode';
import MapperVisualization from './components/MapperVisualization';
import { initializeWasm, createTDAEngine, isWasmReady } from './wasmLoader';
import './App.css';

function App() {
  const [points, setPoints] = useState([]);
  const [filtrationValue, setFiltrationValue] = useState(0.3);
  const [persistenceData, setPersistenceData] = useState(null);
  const [mapperData, setMapperData] = useState(null);
  const [isComputing, setIsComputing] = useState(false);
  const [wasmLoaded, setWasmLoaded] = useState(false);
  const [tdaEngine, setTdaEngine] = useState(null);
  const [activeTab, setActiveTab] = useState('persistence');

  // Initialize WASM on component mount
  useEffect(() => {
    const loadWasm = async () => {
      try {
        const success = await initializeWasm();
        if (success) {
          const engine = createTDAEngine();
          setTdaEngine(engine);
          setWasmLoaded(true);
          console.log('TDA Engine ready');
        } else {
          console.warn('WASM failed to load, using mock computation');
        }
      } catch (error) {
        console.error('WASM initialization error:', error);
        console.warn('Falling back to mock computation');
      }
    };
    
    loadWasm();
  }, []);

  // Generate some sample data
  const generateSampleData = (type) => {
    let newPoints = [];
    
    switch (type) {
      case 'circle':
        for (let i = 0; i < 20; i++) {
          const angle = (2 * Math.PI * i) / 20;
          const radius = 0.3;
          const noise = (Math.random() - 0.5) * 0.05;
          newPoints.push({
            x: 0.5 + (radius + noise) * Math.cos(angle),
            y: 0.5 + (radius + noise) * Math.sin(angle),
            id: Date.now() + i
          });
        }
        break;
      case 'clusters':
        // Create two clusters
        const clusters = [
          { centerX: 0.3, centerY: 0.3 },
          { centerX: 0.7, centerY: 0.7 }
        ];
        clusters.forEach((cluster, clusterIndex) => {
          for (let i = 0; i < 15; i++) {
            newPoints.push({
              x: cluster.centerX + (Math.random() - 0.5) * 0.15,
              y: cluster.centerY + (Math.random() - 0.5) * 0.15,
              id: Date.now() + clusterIndex * 20 + i
            });
          }
        });
        break;
      case 'random':
        for (let i = 0; i < 25; i++) {
          newPoints.push({
            x: Math.random(),
            y: Math.random(),
            id: Date.now() + i
          });
        }
        break;
      default:
        break;
    }
    
    setPoints(newPoints);
  };

  // Persistence computation - uses WASM when available, falls back to mock
  const computePersistence = useCallback(() => {
    if (points.length < 3) {
      setPersistenceData(null);
      return;
    }

    setIsComputing(true);
    
    if (wasmLoaded && tdaEngine) {
      // Use WASM TDA engine
      try {
        // Set points in the TDA engine
        const pointsArray = points.map(p => [p.x, p.y]);
        tdaEngine.set_points(pointsArray);
        
        // Compute Vietoris-Rips complex
        tdaEngine.compute_vietoris_rips(filtrationValue);
        
        // Compute persistence
        const persistenceIntervals = tdaEngine.compute_persistence();
        
        const wasmData = {
          pairs: persistenceIntervals.map(interval => ({
            birth: interval.birth,
            death: interval.death,
            dimension: interval.dimension
          })),
          filtrationValue: filtrationValue
        };
        
        setPersistenceData(wasmData);
        setIsComputing(false);
        console.log('Used WASM computation');
      } catch (error) {
        console.error('WASM computation failed, falling back to mock:', error);
        // Fall back to mock computation
        mockComputation();
      }
    } else {
      // Fall back to mock computation
      mockComputation();
    }
    
    function mockComputation() {
      setTimeout(() => {
        const mockData = {
          pairs: points.slice(0, Math.min(10, points.length)).map((_, i) => ({
            birth: Math.random() * filtrationValue * 0.5,
            death: filtrationValue * (0.6 + Math.random() * 0.4),
            dimension: Math.random() < 0.7 ? 0 : 1
          })),
          filtrationValue: filtrationValue
        };
        
        setPersistenceData(mockData);
        setIsComputing(false);
        console.log('Used mock computation');
      }, 500);
    }
  }, [points, filtrationValue, wasmLoaded, tdaEngine]);

  // Mapper computation - creates network representation of data
  const computeMapper = useCallback(() => {
    if (points.length < 5) {
      setMapperData(null);
      return;
    }

    // Mock Mapper computation - creates a network from point clusters
    const numClusters = Math.min(8, Math.max(3, Math.floor(points.length / 3)));
    const nodes = [];
    const links = [];

    // Create nodes (clusters)
    for (let i = 0; i < numClusters; i++) {
      const centerPoint = points[Math.floor(Math.random() * points.length)];
      const clusterSize = Math.floor(Math.random() * 5) + 2;
      
      nodes.push({
        id: `cluster_${i}`,
        label: `C${i}`,
        x: centerPoint.x * 600, // Scale to visualization coordinates
        y: centerPoint.y * 400,
        radius: 5 + clusterSize * 2,
        size: clusterSize,
        color: `hsl(${(i * 360) / numClusters}, 70%, 60%)`,
        points: points.slice(i * 2, i * 2 + clusterSize) // Mock cluster membership
      });
    }

    // Create links between nearby clusters
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dist = Math.sqrt(
          Math.pow(nodes[i].x - nodes[j].x, 2) + 
          Math.pow(nodes[i].y - nodes[j].y, 2)
        );
        
        // Connect clusters that are close enough
        if (dist < 150 && Math.random() < 0.6) {
          links.push({
            source: nodes[i].id,
            target: nodes[j].id,
            weight: Math.max(0.1, 1 - dist / 150),
            distance: dist
          });
        }
      }
    }

    setMapperData({ nodes, links });
    console.log('Computed Mapper network:', { nodes: nodes.length, links: links.length });
  }, [points]);

  // Handle point creation/editing
  const handlePointsChange = (newPoints) => {
    setPoints(newPoints);
  };

  // Handle filtration value change
  const handleFiltrationChange = (value) => {
    setFiltrationValue(value);
  };

  // Auto-compute when points or filtration changes
  useEffect(() => {
    if (points.length > 0) {
      computePersistence();
    }
  }, [computePersistence]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Topological Data Analysis Visualizer</h1>
        <p>Interactive exploration of persistent homology</p>
      </header>
      
      <div className="app-content">
        <div className="controls">
          <h3>Sample Data</h3>
          <div className="button-group">
            <button onClick={() => generateSampleData('circle')}>Circle</button>
            <button onClick={() => generateSampleData('clusters')}>Clusters</button>
            <button onClick={() => generateSampleData('random')}>Random</button>
            <button onClick={() => setPoints([])}>Clear</button>
          </div>
          
          <div className="filtration-control">
            <label htmlFor="filtration">
              Filtration Value: {filtrationValue.toFixed(3)}
            </label>
            <input
              id="filtration"
              type="range"
              min="0.1"
              max="1.0"
              step="0.01"
              value={filtrationValue}
              onChange={(e) => handleFiltrationChange(parseFloat(e.target.value))}
            />
          </div>
          
          <div className="info">
            <p>Points: {points.length}</p>
            <p>Status: {isComputing ? 'Computing...' : 'Ready'}</p>
            <p>Engine: {wasmLoaded ? 'WASM' : 'Mock'}</p>
          </div>
        </div>
        
        <div className="visualization-container">
          <div className="tab-controls">
            <button 
              className={activeTab === 'persistence' ? 'tab-button active' : 'tab-button'}
              onClick={() => setActiveTab('persistence')}
            >
              Persistence Analysis
            </button>
            <button 
              className={activeTab === 'mapper' ? 'tab-button active' : 'tab-button'}
              onClick={() => setActiveTab('mapper')}
            >
              Mapper Network
            </button>
          </div>

          {activeTab === 'persistence' && (
            <div className="visualization-grid">
              <div className="viz-panel">
                <h3>Point Cloud</h3>
                <PointCloudCanvas
                  points={points}
                  onPointsChange={handlePointsChange}
                  filtrationValue={filtrationValue}
                />
              </div>
              
              <div className="viz-panel">
                <h3>Persistence Diagram</h3>
                <PersistenceDiagram
                  persistenceData={persistenceData}
                  filtrationValue={filtrationValue}
                />
              </div>
              
              <div className="viz-panel">
                <h3>Persistence Barcode</h3>
                <PersistenceBarcode
                  persistenceData={persistenceData}
                  filtrationValue={filtrationValue}
                />
              </div>
            </div>
          )}

          {activeTab === 'mapper' && (
            <div className="mapper-container">
              <div className="viz-panel mapper-main">
                <MapperVisualization
                  mapperData={mapperData}
                  width={800}
                  height={600}
                />
              </div>
              <div className="viz-panel mapper-side">
                <h3>Point Cloud Input</h3>
                <PointCloudCanvas
                  points={points}
                  onPointsChange={handlePointsChange}
                  filtrationValue={filtrationValue}
                  width={300}
                  height={300}
                />
                <button 
                  onClick={computeMapper}
                  disabled={points.length < 5}
                  className="compute-button"
                >
                  Compute Mapper Network
                </button>
                {mapperData && (
                  <div className="mapper-stats">
                    <p>Nodes: {mapperData.nodes?.length || 0}</p>
                    <p>Edges: {mapperData.links?.length || 0}</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
      
      <footer className="app-footer">
        <p>Built with React + D3.js | TDA Engine: Rust/WebAssembly</p>
      </footer>
    </div>
  );
}

export default App;
