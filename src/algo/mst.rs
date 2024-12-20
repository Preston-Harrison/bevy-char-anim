//-------------------------------------------------------
// Minimum Spanning Tree (Kruskal's Algorithm)
//-------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub weight: f32,
}

pub fn kruskal_mst(edges: Vec<Edge>, num_nodes: usize) -> Vec<Edge> {
    let mut sorted_edges = edges;
    sorted_edges.sort_by(|e1, e2| e1.weight.partial_cmp(&e2.weight).unwrap());

    let mut parent: Vec<usize> = (0..num_nodes).collect();
    let mut rank = vec![0; num_nodes];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) -> bool {
        let root_x = find(parent, x);
        let root_y = find(parent, y);

        if root_x != root_y {
            if rank[root_x] > rank[root_y] {
                parent[root_y] = root_x;
            } else if rank[root_x] < rank[root_y] {
                parent[root_x] = root_y;
            } else {
                parent[root_y] = root_x;
                rank[root_x] += 1;
            }
            true
        } else {
            false
        }
    }

    let mut mst = Vec::new();

    for edge in sorted_edges {
        if union(&mut parent, &mut rank, edge.a, edge.b) {
            mst.push(edge);
        }
    }

    mst
}
