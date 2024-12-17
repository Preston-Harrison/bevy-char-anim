use bevy::prelude::Vec3;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Copy, Clone, PartialEq)]
struct NodeCost {
    node: usize,
    f_score: f32,
}

// Implement ordering so that BinaryHeap can sort NodeCost by f_score (lowest first)
impl Eq for NodeCost {}

impl Ord for NodeCost {
    fn cmp(&self, other: &Self) -> Ordering {
        // We want a min-heap based on f_score, so invert comparison
        self.f_score
            .partial_cmp(&other.f_score)
            .unwrap_or(Ordering::Equal)
            .reverse()
    }
}

impl PartialOrd for NodeCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Compute Euclidean distance between two points
fn heuristic(a: Vec3, b: Vec3) -> f32 {
    a.distance(b)
}

/// A* pathfinding
///
/// Given:
/// - points: A list of positions (Vec3) for each node index.
/// - adjacency: A list of adjacency lists, where adjacency[i] is the list of neighbors of node i.
/// - start: Index of the start node
/// - goal: Index of the goal node
///
/// Returns:
/// An Option containing the path as a sequence of node indices, or None if no path is found.
pub fn astar(
    points: &Vec<Vec3>,
    adjacency: &Vec<Vec<usize>>,
    start: usize,
    goal: usize,
) -> Option<Vec<usize>> {
    let n = points.len();
    let mut came_from = vec![None; n];
    let mut g_score = vec![f32::INFINITY; n];
    let mut f_score = vec![f32::INFINITY; n];

    g_score[start] = 0.0;
    f_score[start] = heuristic(points[start], points[goal]);

    let mut open_set = BinaryHeap::new();
    open_set.push(NodeCost {
        node: start,
        f_score: f_score[start],
    });
    let mut in_open_set = HashSet::new();
    in_open_set.insert(start);

    while let Some(NodeCost { node: current, .. }) = open_set.pop() {
        if current == goal {
            // Reconstruct path
            let mut path = Vec::new();
            let mut cur = current;
            while let Some(prev) = came_from[cur] {
                path.push(cur);
                cur = prev;
            }
            path.push(start);
            path.reverse();
            return Some(path);
        }

        in_open_set.remove(&current);

        for &neighbor in &adjacency[current] {
            let tentative_g = g_score[current] + heuristic(points[current], points[neighbor]);
            if tentative_g < g_score[neighbor] {
                came_from[neighbor] = Some(current);
                g_score[neighbor] = tentative_g;
                f_score[neighbor] = tentative_g + heuristic(points[neighbor], points[goal]);
                if !in_open_set.contains(&neighbor) {
                    open_set.push(NodeCost {
                        node: neighbor,
                        f_score: f_score[neighbor],
                    });
                    in_open_set.insert(neighbor);
                }
            }
        }
    }

    None
}
