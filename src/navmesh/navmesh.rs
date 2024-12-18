use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
use bevy::{prelude::*, utils::HashSet};
use bevy_rapier3d::prelude::*;
use std::collections::HashMap;

use crate::utils;

#[derive(Component)]
pub struct NavMeshConstructor;

pub fn setup_navmesh(
    mut commands: Commands,
    mut tracker: Local<HashSet<Entity>>,
    meshes: Query<(Entity, &Mesh3d, &GlobalTransform)>,
    mesh_res: Res<Assets<Mesh>>,
    parents: Query<&Parent>,
    navmesh: Query<&NavMeshConstructor>,
) {
    for (entity, handle, transform) in meshes.iter() {
        if tracker.contains(&entity) {
            continue;
        }
        let Some((navgrid, _)) = utils::find_upwards(entity, &parents, &navmesh) else {
            continue;
        };

        let Some(mesh) = mesh_res.get(&handle.0) else {
            continue;
        };
        tracker.insert(entity);
        let nav = NavMesh::from_mesh(mesh, transform);
        info!("spawning navmesh");
        commands
            .entity(navgrid)
            .insert(nav)
            .insert(Collider::from_bevy_mesh(mesh, &ComputedColliderShape::default()).unwrap())
            .remove::<NavMeshConstructor>();
    }
}

#[derive(Component)]
pub struct NavMesh {
    pub triangles: Vec<[usize; 3]>,
    pub vertex_positions: Vec<Vec3>,
    pub adjacency: Vec<Vec<usize>>,
    pub outside_edges: Vec<[usize; 2]>,
}

impl NavMesh {
    /// Construct a NavMesh from a given Bevy Mesh and its GlobalTransform.
    /// Assumes the mesh is composed of triangles.
    pub fn from_mesh(mesh: &Mesh, transform: &GlobalTransform) -> Self {
        let (mut vertex_positions, mut triangles) = extract_mesh_data(mesh, transform);
        merge::merge_vertices_by_distance(&mut vertex_positions, &mut triangles, 0.1);
        let adjacency = build_adjacency(&triangles, vertex_positions.len());
        let outside_edges = compute_outside_edges(&triangles);

        NavMesh {
            triangles,
            vertex_positions,
            adjacency,
            outside_edges,
        }
    }

    pub fn find_path(&self, from: Vec3, to: Vec3) -> Option<Vec<Vec3>> {
        let path = self.find_vertex_path(from, to)?;
        dbg!(&path);
        let pulled = self.string_pull(&path, 2.0);
        dbg!(&pulled);
        Some(pulled)
    }

    pub fn find_tri_for_point(&self, point: Vec3) -> Option<usize> {
        for (i, tri) in self.triangles.iter().enumerate() {
            if point_in_triangle(
                point,
                self.vertex_positions[tri[0] as usize],
                self.vertex_positions[tri[1] as usize],
                self.vertex_positions[tri[2] as usize],
            ) {
                return Some(i);
            }
        }
        None
    }

    /// Returns the vertices to be travelled to to reach the goal. If there is a
    /// path, it will have at least [start, goal] as a path, even if they are the
    /// same.
    fn find_vertex_path(&self, start: Vec3, goal: Vec3) -> Option<Vec<Vec3>> {
        // TODO: profile this large allocation.
        let mut adjacency = self.adjacency.clone();
        let mut vertex_positions = self.vertex_positions.clone();

        let start_tri = self.find_tri_for_point(start)?;
        let goal_tri = self.find_tri_for_point(goal)?;

        if start_tri == goal_tri {
            return Some(vec![start, goal]);
        }

        vertex_positions.push(start);
        vertex_positions.push(goal);
        let start_ix = vertex_positions.len() - 2;
        let goal_ix = vertex_positions.len() - 1;
        adjacency.push(vec![]);
        adjacency.push(vec![]);

        for &v in &self.triangles[start_tri] {
            adjacency[start_ix].push(v);
            adjacency[v].push(start_ix);
        }
        for &v in &self.triangles[goal_tri] {
            adjacency[goal_ix].push(v);
            adjacency[v].push(goal_ix);
        }

        astar::vertex_astar(&adjacency, &vertex_positions, start_ix, goal_ix)
            .map(|path| path.into_iter().map(|ix| vertex_positions[ix]).collect())
    }

    /// Simplifies a path of Vec3 positions using the string-pulling algorithm.
    /// Skips redundant points while ensuring the path stays within the navmesh.
    ///
    /// # Arguments:
    /// * `path` - A slice of Vec3 world positions representing the original path.
    /// * `threshold` - The vertical distance tolerance for staying within the navmesh.
    ///
    /// # Returns:
    /// A simplified path as a vector of Vec3 positions.
    pub fn string_pull(&self, path: &[Vec3], threshold: f32) -> Vec<Vec3> {
        let mut simplified_path = Vec::new();

        if path.len() <= 2 {
            return path.to_vec();
        }

        simplified_path.push(path[0]);

        for segments in path[1..].windows(2) {
            let curr = segments[0];
            let next = segments[1];
            let last_anchor = simplified_path.last().unwrap();

            if !self.is_visible(*last_anchor, next, threshold) {
                println!("not visible");
                simplified_path.push(curr);
            } else {
                println!("visible");
            }
        }

        // Ensure the final point is included
        let goal = *path.last().unwrap();
        simplified_path.push(goal);

        simplified_path
    }

    pub fn is_visible(&self, from: Vec3, to: Vec3, threshold: f32) -> bool {
        // Compute the average Y-level of the navmesh to treat it as a "floor."
        let avg_y = self.vertex_positions.iter().map(|v| v.y).sum::<f32>() / self.vertex_positions.len() as f32;
    
        // Check vertical threshold against avg_y of the entire navmesh rather than per-edge planes.
        if (from.y - avg_y).abs() > threshold || (to.y - avg_y).abs() > threshold {
            return false;
        }
    
        let line_start = (from.x, from.z);
        let line_end = (to.x, to.z);
    
        // If the line intersects any outside edge in the xz-plane, it's not visible.
        for &[v1, v2] in &self.outside_edges {
            let edge_start = (self.vertex_positions[v1].x, self.vertex_positions[v1].z);
            let edge_end = (self.vertex_positions[v2].x, self.vertex_positions[v2].z);
    
            if line_segments_intersect(line_start, line_end, edge_start, edge_end) {
                return false;
            }
        }
    
        true
    }
}

/// Determines if two line segments intersect in the xz-plane (2D).
fn line_segments_intersect(
    (p1x, p1z): (f32, f32),
    (p2x, p2z): (f32, f32),
    (q1x, q1z): (f32, f32),
    (q2x, q2z): (f32, f32),
) -> bool {
    fn orientation(p: (f32, f32), q: (f32, f32), r: (f32, f32)) -> i32 {
        let val = (q.1 - p.1) * (r.0 - q.0) - (q.0 - p.0) * (r.1 - q.1);
        if val.abs() < f32::EPSILON {
            0
        } else if val > 0.0 {
            1
        } else {
            -1
        }
    }

    fn on_segment(p: (f32, f32), q: (f32, f32), r: (f32, f32)) -> bool {
        q.0 >= p.0.min(r.0) && q.0 <= p.0.max(r.0) && q.1 >= p.1.min(r.1) && q.1 <= p.1.max(r.1)
    }

    let o1 = orientation((p1x, p1z), (p2x, p2z), (q1x, q1z));
    let o2 = orientation((p1x, p1z), (p2x, p2z), (q2x, q2z));
    let o3 = orientation((q1x, q1z), (q2x, q2z), (p1x, p1z));
    let o4 = orientation((q1x, q1z), (q2x, q2z), (p2x, p2z));

    // General case
    if o1 != o2 && o3 != o4 {
        return true;
    }

    // Special cases
    if o1 == 0 && on_segment((p1x, p1z), (q1x, q1z), (p2x, p2z)) {
        return true;
    }
    if o2 == 0 && on_segment((p1x, p1z), (q2x, q2z), (p2x, p2z)) {
        return true;
    }
    if o3 == 0 && on_segment((q1x, q1z), (p1x, p1z), (q2x, q2z)) {
        return true;
    }
    if o4 == 0 && on_segment((q1x, q1z), (p2x, p2z), (q2x, q2z)) {
        return true;
    }

    false
}

/// Checks if the segment from `p1` to `p2` remains close to the plane of the edge (v1, v2).
fn within_plane_threshold(p1: Vec3, p2: Vec3, v1: Vec3, v2: Vec3, threshold: f32) -> bool {
    // Calculate the plane normal from edge (v1 to v2) and an arbitrary perpendicular vector
    let edge = v2 - v1;
    let up = Vec3::Y;
    let plane_normal = edge.cross(up).normalize();

    // Plane equation: N . (P - V) = 0
    let plane_d = -plane_normal.dot(v1);

    // Compute distances of p1 and p2 from the plane
    let distance_from_plane = |point: Vec3| plane_normal.dot(point) + plane_d;

    let d1 = distance_from_plane(p1).abs();
    let d2 = distance_from_plane(p2).abs();

    d1 <= threshold && d2 <= threshold
}

fn compute_outside_edges(triangles: &[[usize; 3]]) -> Vec<[usize; 2]> {
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

    // Helper function to ensure edges have consistent ordering
    let normalize_edge = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };

    // Count the occurrences of each edge
    for triangle in triangles {
        for i in 0..3 {
            let edge = normalize_edge(triangle[i], triangle[(i + 1) % 3]);
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    // Collect edges that appear only once (boundary edges)
    edge_count
        .into_iter()
        .filter(|&(_, count)| count == 1)
        .map(|(edge, _)| [edge.0, edge.1])
        .collect()
}

fn point_in_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    // Vectors from the triangle vertices to the test point
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;

    // Compute dot products
    let dot00 = v0.dot(v0);
    let dot01 = v0.dot(v1);
    let dot02 = v0.dot(v2);
    let dot11 = v1.dot(v1);
    let dot12 = v1.dot(v2);

    // Compute barycentric coordinates
    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    // Check if point is inside the triangle
    (u >= 0.0) && (v >= 0.0) && (u + v <= 1.0)
}

/// Merges vertices that are closer than `threshold` distance apart.
/// Updates both `vertices` and `indices` so that duplicate/close vertices are eliminated.
fn merge_vertices_by_distance(vertices: &mut Vec<Vec3>, indices: &mut Vec<u32>, threshold: f32) {
    let mut unique_vertices = Vec::new();
    let mut remap = vec![0u32; vertices.len()];

    // Naive O(n^2) approach: For each vertex, try to find a close-enough match among unique vertices.
    // For large meshes, consider spatial partitioning or k-d tree for efficiency.
    'outer: for (i, &v) in vertices.iter().enumerate() {
        for (uidx, &uvert) in unique_vertices.iter().enumerate() {
            if v.distance(uvert) < threshold {
                // Found a close vertex, remap this one
                remap[i] = uidx as u32;
                continue 'outer;
            }
        }
        // No close vertex found, add this as a unique vertex
        remap[i] = unique_vertices.len() as u32;
        unique_vertices.push(v);
    }

    // Now remap all indices
    for i in indices.iter_mut() {
        *i = remap[*i as usize];
    }

    *vertices = unique_vertices;
}

fn extract_mesh_data(mesh: &Mesh, transform: &GlobalTransform) -> (Vec<Vec3>, Vec<[usize; 3]>) {
    let vertices = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(verts)) => verts,
        _ => panic!("mesh positions are not Float32x3 format"),
    };

    let triangles = match mesh.indices() {
        Some(Indices::U32(v)) => v
            .chunks(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect(),
        Some(Indices::U16(v)) => v
            .chunks(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect(),
        None => panic!("mesh has no indices"),
    };

    let world_vertices: Vec<Vec3> = vertices
        .iter()
        .map(|v| transform.transform_point(Vec3::new(v[0], v[1], v[2])))
        .collect();

    (world_vertices, triangles)
}

/// Build adjacency list for vertices, where each vertex index maps to its adjacent vertices.
/// `triangles` is a slice of triangles, where each triangle is an array of three vertex indices.
/// `vertex_count` is the total number of unique vertices.
fn build_adjacency(triangles: &[[usize; 3]], vertex_count: usize) -> Vec<Vec<usize>> {
    let mut adjacency = vec![vec![]; vertex_count]; // Adjacency list for each vertex

    for triangle in triangles {
        // For each triangle, connect its vertices to one another
        let vertices = [triangle[0], triangle[1], triangle[2]];

        for i in 0..3 {
            for j in 0..3 {
                if i != j && !adjacency[vertices[i]].contains(&vertices[j]) {
                    adjacency[vertices[i]].push(vertices[j]);
                }
            }
        }
    }

    adjacency
}

fn sort_edge_indices(edge: (u32, u32)) -> (u32, u32) {
    if edge.0 < edge.1 {
        edge
    } else {
        (edge.1, edge.0)
    }
}

mod astar {
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
}

mod merge {
    use bevy::prelude::*;

    /// A simple union-find data structure
    struct UnionFind {
        parent: Vec<usize>,
        rank: Vec<usize>,
    }

    impl UnionFind {
        fn new(size: usize) -> Self {
            Self {
                parent: (0..size).collect(),
                rank: vec![0; size],
            }
        }

        fn find(&mut self, x: usize) -> usize {
            if self.parent[x] != x {
                self.parent[x] = self.find(self.parent[x]);
            }
            self.parent[x]
        }

        fn union(&mut self, a: usize, b: usize) {
            let root_a = self.find(a);
            let root_b = self.find(b);

            if root_a != root_b {
                // Union by rank
                if self.rank[root_a] < self.rank[root_b] {
                    self.parent[root_a] = root_b;
                } else if self.rank[root_a] > self.rank[root_b] {
                    self.parent[root_b] = root_a;
                } else {
                    self.parent[root_b] = root_a;
                    self.rank[root_a] += 1;
                }
            }
        }
    }

    /// Merge vertices that are closer than `threshold`. After merging,
    /// the triangles' vertex indices are updated to reflect the merged vertices.
    pub fn merge_vertices_by_distance(
        vertices: &mut Vec<Vec3>,
        triangles: &mut Vec<[usize; 3]>,
        threshold: f32,
    ) {
        let len = vertices.len();
        if len == 0 {
            return;
        }

        // Create a union-find structure for all vertices
        let mut uf = UnionFind::new(len);

        // O(n²) approach: check all pairs of vertices and union if within threshold
        // For large meshes, consider spatial partitioning to reduce complexity.
        for i in 0..len {
            for j in (i + 1)..len {
                if vertices[i].distance(vertices[j]) < threshold {
                    uf.union(i, j);
                }
            }
        }

        // Each union-find set of vertices should collapse into a single vertex.
        // We'll pick one representative vertex per set.
        // Create a mapping from old vertex index -> representative set index
        let mut representative_map = Vec::with_capacity(len);
        for i in 0..len {
            representative_map.push(uf.find(i));
        }

        // Now we need to compress these sets into a minimal set of unique vertices.
        // unique_roots: map from root -> new_index
        let mut unique_roots = Vec::new();
        let mut root_to_new = Vec::with_capacity(len);
        root_to_new.resize(len, usize::MAX);

        for &root in &representative_map {
            if root_to_new[root] == usize::MAX {
                root_to_new[root] = unique_roots.len();
                unique_roots.push(root);
            }
        }

        // Build the new vertex array
        let mut new_vertices = Vec::with_capacity(unique_roots.len());
        for &root in &unique_roots {
            // The representative vertex can just be the original vertex at the root
            // Alternatively, you could average all vertices in the set for a more "smooth" merge.
            new_vertices.push(vertices[root]);
        }

        // Update triangles
        for tri in triangles.iter_mut() {
            for idx in tri.iter_mut() {
                let old_idx = *idx as usize;
                let root = representative_map[old_idx];
                let new_idx = root_to_new[root];
                *idx = new_idx;
            }
        }

        *vertices = new_vertices;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_l_shaped_navmesh() -> NavMesh {
        // Define a 3x3 grid of vertices in xz-plane at y=0:
        //   V6(0,2)---V7(1,2)---V8(2,2)
        //    | \       | \       |
        //   V3(0,1)---V4(1,1)---V5(2,1)
        //    | \       | 
        //   V0(0,0)---V1(1,0)   V2(2,0) <- This corner is missing a square, forming an "L"
        //
        // We use x for horizontal and z for vertical coordinate. All y=0.
        let vertex_positions = vec![
            Vec3::new(0.0, 0.0, 0.0), // V0
            Vec3::new(1.0, 0.0, 0.0), // V1
            Vec3::new(2.0, 0.0, 0.0), // V2
            Vec3::new(0.0, 0.0, 1.0), // V3
            Vec3::new(1.0, 0.0, 1.0), // V4
            Vec3::new(2.0, 0.0, 1.0), // V5
            Vec3::new(0.0, 0.0, 2.0), // V6
            Vec3::new(1.0, 0.0, 2.0), // V7
            Vec3::new(2.0, 0.0, 2.0), // V8
        ];

        // Triangulate the "L" shape. We have:
        // Top-left square: (V6, V7, V3, V4)
        //    Triangles: (6,7,4) and (6,4,3)
        // Top-right square: (V7, V8, V4, V5)
        //    Triangles: (7,8,5) and (7,5,4)
        // Bottom-left square: (V3, V4, V0, V1)
        //    Triangles: (3,4,1) and (3,1,0)
        //
        // Missing bottom-right square (V4,V5,V1,V2) creates the "L" shape.

        let triangles = vec![
            [6, 7, 4], // top-left square, triangle 1
            [6, 4, 3], // top-left square, triangle 2
            [7, 8, 5], // top-right square, triangle 1
            [7, 5, 4], // top-right square, triangle 2
            [3, 4, 1], // bottom-left square, triangle 1
            [3, 1, 0], // bottom-left square, triangle 2
        ];

        let adjacency = build_adjacency(&triangles, vertex_positions.len());
        let outside_edges = compute_outside_edges(&triangles);

        NavMesh {
            triangles,
            vertex_positions,
            adjacency,
            outside_edges,
        }
    }

    #[test]
    fn test_line_visible_within_mesh() {
        let navmesh = create_l_shaped_navmesh();

        // A line entirely within the top row of the L
        let from = Vec3::new(0.5, 0.0, 1.9);
        let to = Vec3::new(1.5, 0.0, 1.9);
        assert!(navmesh.is_visible(from, to, 0.1), "Line should be visible inside the mesh.");
    }

    #[test]
    fn test_line_not_visible_outside_mesh() {
        let navmesh = create_l_shaped_navmesh();

        // A line that attempts to cross the missing bottom-right square region.
        // From somewhere in bottom-left area to top-right area, diagonally crossing outside the L.
        let from = Vec3::new(0.5, 0.0, 0.5); // Within bottom-left square
        let to = Vec3::new(1.5, 0.0, 1.5);   // Approaching the top-right area
        assert!(!navmesh.is_visible(from, to, 0.1), "Line should not be visible as it crosses outside the L shape.");
    }

    #[test]
    fn test_line_just_inside_corner() {
        let navmesh = create_l_shaped_navmesh();

        // A line near the inner bend of the L, staying just inside the mesh.
        let from = Vec3::new(0.9, 0.0, 0.9);
        let to = Vec3::new(0.9, 0.0, 1.1);
        assert!(navmesh.is_visible(from, to, 0.1), "Line should be visible close to the inner corner of the L.");
    }
}
