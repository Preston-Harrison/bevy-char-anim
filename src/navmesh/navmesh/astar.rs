use std::{cmp::Ordering, collections::BinaryHeap};

use bevy::{prelude::*, utils::HashMap};

/// Node struct to hold A* frontier data
#[derive(Copy, Clone, PartialEq)]
struct Node {
    cost: f32,
    index: usize,
}

/// Implement ordering for the BinaryHeap to prioritize nodes with lower cost
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Run A* on the polygon graph. Returns the vertex path from start to goal,
/// where start and goal are indices into the vertex array. Vertex connections
/// is a list of connected vertices per vertex. Vertex world space contains
/// the world coordinates per vertex.
///
/// Heuristic: Euclidean distance between vertices.
pub fn vertex_astar(
    vertex_connections: &[Vec<usize>],
    vertex_world_space: &[Vec3],
    start: usize,
    goal: usize,
) -> Option<Vec<usize>> {
    let mut open_set = BinaryHeap::new();
    let mut came_from: HashMap<usize, usize> = HashMap::new();
    let mut g_score: HashMap<usize, f32> = HashMap::new();
    let mut f_score: HashMap<usize, f32> = HashMap::new();

    open_set.push(Node {
        index: start,
        cost: 0.0,
    });
    g_score.insert(start, 0.0);
    f_score.insert(
        start,
        heuristic(vertex_world_space[start], vertex_world_space[goal]),
    );

    while let Some(Node { index: current, .. }) = open_set.pop() {
        if current == goal {
            return Some(reconstruct_path(&came_from, current));
        }

        for &neighbor in &vertex_connections[current] {
            let tentative_g_score = g_score.get(&current).unwrap_or(&f32::INFINITY)
                + vertex_world_space[current].distance(vertex_world_space[neighbor]);

            if tentative_g_score < *g_score.get(&neighbor).unwrap_or(&f32::INFINITY) {
                came_from.insert(neighbor, current);
                g_score.insert(neighbor, tentative_g_score);
                let f_score_value = tentative_g_score
                    + heuristic(vertex_world_space[neighbor], vertex_world_space[goal]);
                f_score.insert(neighbor, f_score_value);
                open_set.push(Node {
                    index: neighbor,
                    cost: f_score_value,
                });
            }
        }
    }

    None
}

/// Heuristic: Euclidean distance between two points
fn heuristic(a: Vec3, b: Vec3) -> f32 {
    a.distance(b)
}

/// Reconstruct the path by backtracking through the `came_from` map
fn reconstruct_path(came_from: &HashMap<usize, usize>, mut current: usize) -> Vec<usize> {
    let mut path = vec![current];
    while let Some(&parent) = came_from.get(&current) {
        current = parent;
        path.push(current);
    }
    path.reverse();
    path
}
