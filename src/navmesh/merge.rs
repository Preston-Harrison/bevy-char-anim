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
