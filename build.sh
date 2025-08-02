#!/bin/bash

# Build script for TDA Visualizer

echo "🦀 Building Rust/WASM core..."
cd tda_rust_core || exit

# Install wasm-pack if not available
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Build the Rust code to WASM
wasm-pack build --target bundler --out-dir pkg

echo "📦 Installing JavaScript dependencies..."
cd ../web_visualizer || exit
npm install

echo "🚀 Starting development server..."
npm start
