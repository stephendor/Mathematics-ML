#!/bin/bash

# Quick start script for development

echo "🚀 Starting TDA Visualizer in development mode..."

cd /home/stephen-dorman/dev/Mathematics-ML/web_visualizer || exit

# Check if WASM is built
if [ ! -d "../tda_rust_core/pkg" ]; then
    echo "⚠️  WASM package not found. Building Rust core first..."
    cd ../tda_rust_core || exit
    /home/stephen-dorman/.cargo/bin/wasm-pack build --target bundler --out-dir pkg
    cd ../web_visualizer || exit
fi

echo "📱 Starting React development server..."
npm start
