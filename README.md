# Interactive Topological Data Analysis Visualizer

A high-performance, real-time TDA application combining Rust/WASM computation with React/D3.js visualization.

## 🌟 Features

- **Real-time Persistent Homology**: Interactive computation of persistence diagrams and barcodes
- **Multiple Visualizations**: Point clouds, simplicial complexes, persistence diagrams, and barcodes
- **High Performance**: Rust/WASM backend for efficient TDA algorithms
- **Interactive Interface**: Click-to-add points, adjustable filtration parameters
- **Educational**: Perfect for learning and teaching topological data analysis concepts

## 🚀 Quick Start

```bash
# Clone and run
git clone <repository-url>
cd Mathematics-ML
./build.sh
```

This will:
1. Build the Rust/WASM TDA core
2. Install JavaScript dependencies  
3. Start the development server

## 🏗️ Architecture

```
Mathematics-ML/
├── tda_rust_core/          # High-performance Rust TDA algorithms
│   ├── src/lib.rs          # Core TDA implementation
│   └── Cargo.toml          # Rust dependencies
├── web_visualizer/         # React frontend with D3.js visualizations
│   ├── src/
│   │   ├── components/     # Reusable visualization components
│   │   ├── App.js          # Main application
│   │   └── App.css         # Styling
└── planning_docs/          # Research and project documentation
```

## 🔬 TDA Implementation

### Core Algorithms
- **Vietoris-Rips Complex Construction**: Efficient simplicial complex generation
- **Persistent Homology**: 0-dimensional (connected components) computation  
- **Union-Find**: Optimized component tracking
- **Distance Matrix**: Euclidean distance computation

### Visualization Components
- **PointCloudCanvas**: Interactive point cloud with simplicial complex overlay
- **PersistenceDiagram**: Birth-death scatter plot with dimensional coloring
- **PersistenceBarcode**: Interval representation of topological features

## 🎯 Portfolio Impact

This project demonstrates:

1. **Technical Depth**: Advanced mathematical concepts (algebraic topology) implemented efficiently
2. **Modern Stack**: Rust/WASM + React + D3.js integration
3. **User Experience**: Real-time, interactive mathematical visualization
4. **Educational Value**: Makes complex TDA concepts accessible and engaging
5. **Performance**: High-speed computation suitable for larger datasets

## 🛠️ Development

### Prerequisites
- Rust (latest stable)
- Node.js 14+
- wasm-pack

### Build Commands

```bash
# Build Rust core only
cd tda_rust_core
wasm-pack build --target bundler

# Build frontend only  
cd web_visualizer
npm install && npm start

# Full build
./build.sh
```

## 📊 Usage Examples

1. **Generate Sample Data**: Use built-in generators for circles, clusters, or random points
2. **Interactive Drawing**: Click to add points, shift+click to remove
3. **Adjust Filtration**: Use the slider to see how topology changes with scale
4. **Explore Results**: Hover over persistence features for detailed information

## 🎨 Visualization Features

- **Multi-dimensional Homology**: Color-coded by dimension (H₀, H₁, H₂)
- **Interactive Tooltips**: Detailed feature information on hover
- **Real-time Updates**: Automatic recomputation as data changes
- **Responsive Design**: Works on desktop and tablet devices

## 📈 Future Enhancements

- **Higher-dimensional Persistence**: Extend to H₂ and beyond
- **Multiple Distance Metrics**: Support for non-Euclidean distances
- **Data Import**: Load real datasets (CSV, JSON)
- **3D Visualization**: Three.js integration for spatial data
- **Mapper Algorithm**: Additional TDA method implementation

## 🎓 Educational Applications

Perfect for:
- **Topology Courses**: Interactive demonstrations of persistent homology
- **Data Science Workshops**: Real-world applications of TDA
- **Research Presentations**: Clear visualization of topological features
- **Self-Learning**: Hands-on exploration of TDA concepts

## 📚 References

Based on computational topology theory and modern TDA algorithms. See `planning_docs/` for detailed mathematical background and project roadmap.