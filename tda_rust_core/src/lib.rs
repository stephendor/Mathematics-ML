use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use ndarray::Array2;

// Enable console.log in Rust
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PersistencePair {
    pub birth: f64,
    pub death: f64,
    pub dimension: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
    pub max_filtration: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VietorisRipsComplex {
    pub simplices: Vec<Vec<usize>>,
    pub filtration_values: Vec<f64>,
}

#[wasm_bindgen]
pub struct TDAEngine {
    points: Vec<Point2D>,
    distance_matrix: Option<Array2<f64>>,
}

#[wasm_bindgen]
impl TDAEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> TDAEngine {
        console_log!("TDA Engine initialized");
        TDAEngine {
            points: Vec::new(),
            distance_matrix: None,
        }
    }

    #[wasm_bindgen]
    pub fn set_points(&mut self, points_js: &JsValue) -> Result<(), JsValue> {
        let points: Vec<Point2D> = serde_wasm_bindgen::from_value(points_js.clone())?;
        self.points = points;
        self.compute_distance_matrix();
        console_log!("Set {} points", self.points.len());
        Ok(())
    }

    fn compute_distance_matrix(&mut self) {
        let n = self.points.len();
        let mut distances = Array2::zeros((n, n));
        
        for i in 0..n {
            for j in i..n {
                let dist = self.euclidean_distance(&self.points[i], &self.points[j]);
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }
        
        self.distance_matrix = Some(distances);
        console_log!("Computed distance matrix {}x{}", n, n);
    }

    fn euclidean_distance(&self, p1: &Point2D, p2: &Point2D) -> f64 {
        ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt()
    }

    #[wasm_bindgen]
    pub fn compute_vietoris_rips(&self, max_epsilon: f64, max_dimension: usize) -> Result<JsValue, JsValue> {
        let distance_matrix = self.distance_matrix.as_ref()
            .ok_or_else(|| JsValue::from_str("No distance matrix computed"))?;
        
        console_log!("Computing Vietoris-Rips complex with max_epsilon={}, max_dim={}", max_epsilon, max_dimension);
        
        // Simple Vietoris-Rips construction
        let mut simplices = Vec::new();
        let mut filtration_values = Vec::new();
        let n = self.points.len();
        
        // Add vertices (0-simplices)
        for i in 0..n {
            simplices.push(vec![i]);
            filtration_values.push(0.0);
        }
        
        // Add edges (1-simplices)
        for i in 0..n {
            for j in (i+1)..n {
                let dist = distance_matrix[[i, j]];
                if dist <= max_epsilon {
                    simplices.push(vec![i, j]);
                    filtration_values.push(dist);
                }
            }
        }
        
        // Add triangles (2-simplices) if max_dimension >= 2
        if max_dimension >= 2 {
            for i in 0..n {
                for j in (i+1)..n {
                    for k in (j+1)..n {
                        let max_edge_dist = distance_matrix[[i, j]]
                            .max(distance_matrix[[j, k]])
                            .max(distance_matrix[[i, k]]);
                        
                        if max_edge_dist <= max_epsilon {
                            simplices.push(vec![i, j, k]);
                            filtration_values.push(max_edge_dist);
                        }
                    }
                }
            }
        }
        
        let complex = VietorisRipsComplex {
            simplices,
            filtration_values,
        };
        
        console_log!("Built complex with {} simplices", complex.simplices.len());
        
        serde_wasm_bindgen::to_value(&complex).map_err(|e| {
            JsValue::from_str(&format!("Serialization error: {}", e))
        })
    }

    #[wasm_bindgen]
    pub fn compute_persistence(&self, max_epsilon: f64) -> Result<JsValue, JsValue> {
        console_log!("Computing persistent homology");
        
        // Simplified persistence computation
        // This is a basic implementation - in practice, you'd use a more sophisticated algorithm
        let distance_matrix = self.distance_matrix.as_ref()
            .ok_or_else(|| JsValue::from_str("No distance matrix computed"))?;
        
        let mut pairs = Vec::new();
        let n = self.points.len();
        
        // Track connected components (0-dimensional homology)
        let component_birth_times = vec![0.0; n];
        let mut union_find = UnionFind::new(n);
        
        // Collect all edges with their distances
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for j in (i+1)..n {
                let dist = distance_matrix[[i, j]];
                if dist <= max_epsilon {
                    edges.push((i, j, dist));
                }
            }
        }
        
        // Sort edges by distance
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        
        // Process edges to find when components merge
        for (i, j, dist) in edges {
            let root_i = union_find.find(i);
            let root_j = union_find.find(j);
            
            if root_i != root_j {
                // Components merge - the younger component dies
                let birth_i = component_birth_times[root_i];
                let birth_j = component_birth_times[root_j];
                
                if birth_i > birth_j {
                    pairs.push(PersistencePair {
                        birth: birth_i,
                        death: dist,
                        dimension: 0,
                    });
                } else {
                    pairs.push(PersistencePair {
                        birth: birth_j,
                        death: dist,
                        dimension: 0,
                    });
                }
                
                union_find.union(i, j);
            }
        }
        
        // Add infinite persistence pair for the surviving component
        pairs.push(PersistencePair {
            birth: 0.0,
            death: f64::INFINITY,
            dimension: 0,
        });
        
        let diagram = PersistenceDiagram {
            pairs,
            max_filtration: max_epsilon,
        };
        
        console_log!("Computed {} persistence pairs", diagram.pairs.len());
        
        serde_wasm_bindgen::to_value(&diagram).map_err(|e| {
            JsValue::from_str(&format!("Serialization error: {}", e))
        })
    }
}

// Simple Union-Find data structure for connected components
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    
    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x != root_y {
            if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
    }
}

#[wasm_bindgen]
pub fn add(a: u32, b: u32) -> u32 {
    a + b
}