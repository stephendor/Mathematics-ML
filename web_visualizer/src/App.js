import React, { useState, useEffect } from 'react';
import PointCloudCanvas from './components/PointCloudCanvas';
import PersistenceDiagram from './components/PersistenceDiagram';
import PersistenceBarcode from './components/PersistenceBarcode';
import './App.css';

function App() {
  const [points, setPoints] = useState([]);
  const [filtrationValue, setFiltrationValue] = useState(0.3);
  const [persistenceData, setPersistenceData] = useState(null);
  const [isComputing, setIsComputing] = useState(false);
  const [wasmLoaded, setWasmLoaded] = useState(false);
  const [tdaEngine, setTdaEngine] = useState(null);

  // Load WASM module
  useEffect(() => {
    const loadWasm = async () => {
      try {
        const wasm = await import('../tda_rust_core/pkg');
        const engine = new wasm.TDAEngine();
        setTdaEngine(engine);
        setWasmLoaded(true);
        console.log('WASM module loaded successfully');
      } catch (error) {
        console.error('Failed to load WASM module:', error);
        // For now, we'll work without WASM
        setWasmLoaded(false);
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
        // Three clusters
        const centers = [[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]];
        centers.forEach((center, clusterIdx) => {
          for (let i = 0; i < 8; i++) {
            newPoints.push({
              x: center[0] + (Math.random() - 0.5) * 0.15,
              y: center[1] + (Math.random() - 0.5) * 0.15,
              id: Date.now() + clusterIdx * 100 + i
            });
          }
        });
        break;
      
      case 'random':
        for (let i = 0; i < 25; i++) {
          newPoints.push({
            x: Math.random() * 0.8 + 0.1,
            y: Math.random() * 0.8 + 0.1,
            id: Date.now() + i
          });
        }
        break;
      
      default:
        break;
    }
    
    setPoints(newPoints);
  };

  // Compute persistence (mock implementation when WASM not available)
  const computePersistence = async () => {
    if (points.length < 2) return;
    
    setIsComputing(true);
    
    try {
      if (wasmLoaded && tdaEngine) {
        // Use WASM implementation
        await tdaEngine.set_points(points.map(p => ({ x: p.x, y: p.y })));
        const result = await tdaEngine.compute_persistence(filtrationValue * 2);
        setPersistenceData(result);
      } else {
        // Mock implementation for demonstration
        const mockPersistence = generateMockPersistence(points, filtrationValue);
        setPersistenceData(mockPersistence);
      }
    } catch (error) {
      console.error('Error computing persistence:', error);
    } finally {
      setIsComputing(false);
    }
  };

  // Mock persistence computation for demonstration
  const generateMockPersistence = (pointsData, maxEpsilon) => {
    const pairs = [];
    
    // Mock connected components
    const numComponents = Math.max(1, Math.floor(pointsData.length / 5));
    for (let i = 0; i < numComponents - 1; i++) {
      pairs.push({
        birth: Math.random() * 0.1,
        death: Math.random() * maxEpsilon + 0.1,
        dimension: 0
      });
    }
    
    // One infinite component
    pairs.push({
      birth: 0.0,
      death: Infinity,
      dimension: 0
    });
    
    // Mock 1-dimensional features (loops)
    if (pointsData.length > 10) {
      const numLoops = Math.floor(Math.random() * 3);
      for (let i = 0; i < numLoops; i++) {
        pairs.push({
          birth: Math.random() * maxEpsilon * 0.5 + 0.05,
          death: Math.random() * maxEpsilon * 0.5 + maxEpsilon * 0.6,
          dimension: 1
        });
      }
    }
    
    return {
      pairs,
      max_filtration: maxEpsilon * 2
    };
  };

  // Auto-compute when points or filtration changes
  useEffect(() => {
    if (points.length > 0) {
      const timer = setTimeout(() => {
        computePersistence();
      }, 500); // Debounce
      
      return () => clearTimeout(timer);
    }
  }, [points, filtrationValue, wasmLoaded, tdaEngine]);

  return (
    <div className="App">
      <header className="app-header">
        <h1>Interactive Topological Data Analysis</h1>
        <p>Explore persistent homology through real-time computation and visualization</p>
        {!wasmLoaded && (
          <div className="warning">
            Note: WASM module not loaded. Using mock persistence computation for demonstration.
          </div>
        )}
      </header>

      <div className="main-content">
        <div className="controls-section">
          <div className="sample-data-controls">
            <h3>Sample Data</h3>
            <button onClick={() => generateSampleData('circle')} className="btn">
              Generate Circle
            </button>
            <button onClick={() => generateSampleData('clusters')} className="btn">
              Generate Clusters
            </button>
            <button onClick={() => generateSampleData('random')} className="btn">
              Random Points
            </button>
          </div>

          <div className="filtration-controls">
            <h3>Filtration Parameter</h3>
            <div className="slider-container">
              <label htmlFor="filtration">ε = {filtrationValue.toFixed(3)}</label>
              <input
                id="filtration"
                type="range"
                min="0.01"
                max="0.8"
                step="0.01"
                value={filtrationValue}
                onChange={(e) => setFiltrationValue(parseFloat(e.target.value))}
                className="slider"
              />
            </div>
          </div>

          <div className="info-section">
            <h3>Dataset Info</h3>
            <p>Points: {points.length}</p>
            <p>Status: {isComputing ? 'Computing...' : 'Ready'}</p>
          </div>
        </div>

        <div className="visualization-grid">
          <div className="vis-panel">
            <h3>Point Cloud & Simplicial Complex</h3>
            <PointCloudCanvas 
              points={points}
              onPointsChange={setPoints}
              filtrationValue={filtrationValue}
            />
          </div>

          <div className="vis-panel">
            <h3>Persistence Diagram</h3>
            {persistenceData ? (
              <PersistenceDiagram 
                persistenceData={persistenceData}
                maxFiltration={filtrationValue * 2}
              />
            ) : (
              <div className="placeholder">
                Add some points to see the persistence diagram
              </div>
            )}
          </div>

          <div className="vis-panel full-width">
            <h3>Persistence Barcode</h3>
            {persistenceData ? (
              <PersistenceBarcode 
                persistenceData={persistenceData}
                maxFiltration={filtrationValue * 2}
              />
            ) : (
              <div className="placeholder">
                Add some points to see the persistence barcode
              </div>
            )}
          </div>
        </div>
      </div>

      <footer className="app-footer">
        <p>
          Built with Rust + WASM for high-performance TDA computation and React + D3.js for interactive visualization.
          {wasmLoaded ? ' ⚡ WASM acceleration enabled' : ' 🔧 Running in demo mode'}
        </p>
      </footer>
    </div>
  );
}

export default App;