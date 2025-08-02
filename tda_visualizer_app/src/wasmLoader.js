// Simple WASM loader - disabled for now to avoid import issues
// WASM integration can be re-enabled once import path issues are resolved

export async function initializeWasm() {
  // Return false to use mock computation for now
  console.log('WASM loader: Using mock computation (WASM disabled)');
  return false;
}

export function createTDAEngine() {
  throw new Error('WASM not available. Using mock computation.');
}

export function isWasmReady() {
  return false;
}