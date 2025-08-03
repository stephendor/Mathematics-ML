// Simple mock WASM loader for demo
export async function initializeWasm() {
  console.log('Using mock TDA computation');
  return false;
}

export function createTDAEngine() {
  throw new Error('Using mock computation');
}

export function isWasmReady() {
  return false;
}
