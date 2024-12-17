use avian3d::prelude::*;
use bevy::{color::palettes::css::*, prelude::*, utils::HashSet};
use navmesh::NavMesh;

use crate::utils::{self, freecam::FreeCamera};

pub fn run() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PhysicsPlugins::default())
        .add_plugins(PhysicsDebugPlugin::default())
        .add_plugins(utils::freecam::FreeCameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                setup_navmesh,
                utils::toggle_cursor_grab_with_esc,
                draw_navmesh_path,
            ),
        )
        .run();
}

#[derive(Component)]
struct NavMeshConstructor;

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Spawn the camera.
    commands.spawn((
        Camera3d::default(),
        FreeCamera::new(4.0),
        RayCaster::new(Vec3::ZERO, -Dir3::Z),
        Transform::from_translation(Vec3::splat(6.0)).looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
    ));

    // Spawn the light.
    commands.spawn((
        PointLight {
            intensity: 10_000_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(-4.0, 8.0, 13.0),
    ));

    commands.spawn((
        SceneRoot(asset_server.load("terrain.glb#Scene0")),
        NavMeshConstructor,
    ));
}

fn setup_navmesh(
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
        commands
            .entity(navgrid)
            .insert(nav)
            .remove::<NavMeshConstructor>();
        commands
            .entity(entity)
            .insert(ColliderConstructor::TrimeshFromMesh);
    }
}

/// A Bevy system that allows you to click two points and visualize a path on a navmesh.
/// It uses the camera raycasting setup from your code, as well as bevy gizmos for drawing.
///
/// Steps:
/// 1. Cast a ray from the camera through the mouse cursor.
/// 2. On mouse click, either set the start polygon or the end polygon of the path.
/// 3. Once start and end are chosen, run `navmesh.find_path` to get a path of centroids.
/// 4. Draw gizmos at clicked points and draw a line along the found path.
pub fn draw_navmesh_path(
    mut start_poly: Local<Option<usize>>,
    mut end_poly: Local<Option<usize>>,
    mut path: Local<Option<Vec<Vec3>>>,
    mut centroid_path: Local<Option<Vec<Vec3>>>,
    mut gizmos: Gizmos,
    navmesh_query: Query<&NavMesh>,
    camera_query: Query<(&GlobalTransform, &RayCaster, &RayHits), With<Camera>>,
    mouse: Res<ButtonInput<MouseButton>>,
) {
    let Ok((cam_t, ray, hits)) = camera_query.get_single() else {
        return;
    };
    let Ok(navmesh) = navmesh_query.get_single() else {
        return;
    };

    // for (polygon, others) in navmesh.adjacency.iter().enumerate() {
    //     for other in others {
    //         let start = navmesh.polygons[polygon].centroid;
    //         let end = navmesh.polygons[*other].centroid;
    //         gizmos.arrow(start, end, GREEN_YELLOW);
    //     }
    // }

    let mut hovered_point: Option<Vec3> = None;

    // Check if we have a ray hit in the scene
    if let Some(hit) = hits.iter_sorted().next() {
        // Compute the hit point in world space
        let point =
            cam_t.translation() + ray.origin + (cam_t.rotation() * *ray.direction * hit.distance);

        // Draw a small sphere at the hovered point
        gizmos.sphere(Isometry3d::from_translation(point), 0.2, PURPLE);

        hovered_point = Some(point);
    }

    // On left mouse click, set start or end polygon
    if mouse.just_pressed(MouseButton::Left) {
        // If we have a hovered point, find which polygon it belongs to
        if let Some(hp) = hovered_point {
            let poly = navmesh.find_polygon_for_point(hp);

            if start_poly.is_none() || end_poly.is_some() {
                // Reset if we already had a path or set the start
                *start_poly = poly;
                *end_poly = None;
                *path = None;
            } else if start_poly.is_some() && end_poly.is_none() {
                *end_poly = poly;
            }

            // If we have both start and end, compute a path
            if let (Some(sp), Some(ep)) = (*start_poly, *end_poly) {
                let end_point = navmesh.polygons[ep].centroid;
                let start_point = navmesh.polygons[sp].centroid;
                *path = navmesh.find_path(start_point, end_point);
                *centroid_path = navmesh.find_centroid_path(start_point, end_point);
                if path.is_none() {
                    info!("path is none");
                }
            } else {
                info!("start poly is some: {}", start_poly.is_some());
                info!("end poly is some: {}", end_poly.is_some());
            }
        }
    }

    // Draw start and end polygon indicators
    if let Some(sp) = *start_poly {
        let pos = navmesh.polygons[sp].centroid;
        gizmos.sphere(Isometry3d::from_translation(pos), 0.15, GREEN);
    }
    if let Some(ep) = *end_poly {
        let pos = navmesh.polygons[ep].centroid;
        gizmos.sphere(Isometry3d::from_translation(pos), 0.15, RED);
    }

    // Draw the path if available
    if let Some(ref path_points) = *path {
        for segment in path_points.windows(2) {
            let start = segment[0];
            let end = segment[1];
            gizmos.line(start, end, YELLOW);
        }
    }

    if let Some(ref path_points) = *centroid_path {
        for segment in path_points.windows(2) {
            let start = segment[0];
            let end = segment[1];
            gizmos.line(start, end, RED);
        }
    }
}

mod navmesh {
    use bevy::prelude::*;
    use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
    use std::cmp::Ordering;
    use std::collections::{BinaryHeap, HashMap, HashSet};

    pub struct Polygon {
        pub vertex_indices: [u32; 3],
        pub vertices: [Vec3; 3],
        pub normal: Vec3,
        pub centroid: Vec3,
    }

    /// Represents the navigation mesh as:
    /// - A list of polygons (triangles).
    /// - Adjacency information between polygons.
    #[derive(Component)]
    pub struct NavMesh {
        pub polygons: Vec<Polygon>,
        pub adjacency: Vec<Vec<usize>>, // adjacency[i] lists indices of polygons adjacent to polygon i
    }

    impl NavMesh {
        /// Construct a NavMesh from a given Bevy Mesh and its GlobalTransform.
        /// Assumes the mesh is composed of triangles.
        pub fn from_mesh(mesh: &Mesh, transform: &GlobalTransform) -> Self {
            let (mut vertices, mut indices) = extract_mesh_data(mesh, transform);
            merge_vertices_by_distance(&mut vertices, &mut indices, 0.01);
            let polygons = build_polygons(&vertices, &indices);
            let adjacency = build_adjacency(&polygons);

            NavMesh {
                polygons,
                adjacency,
            }
        }

        pub fn find_polygon_for_point(&self, point: Vec3) -> Option<usize> {
            for (i, poly) in self.polygons.iter().enumerate() {
                if Self::point_in_triangle(point, poly.vertices[0], poly.vertices[1], poly.vertices[2]) {
                    return Some(i);
                }
            }
            None
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

        /// Compute a path between two points on the navmesh by:
        /// 1. Finding their polygons.
        /// 2. Running A* at the polygon level.
        /// The returned path is a list of polygon centroids forming a route.
        pub fn find_centroid_path(&self, start: Vec3, goal: Vec3) -> Option<Vec<Vec3>> {
            let start_poly = self.find_polygon_for_point(start)?;
            let goal_poly = self.find_polygon_for_point(goal)?;

            // If we're already in the goal polygon, return a trivial path
            if start_poly == goal_poly {
                return Some(vec![start, goal]);
            }

            let path_polygons =
                polygon_astar(&self.polygons, &self.adjacency, start_poly, goal_poly)?;

            // Convert polygons to a path via their centroids.
            // Optionally, you could run a funnel algorithm for a smoother path.
            let mut path = Vec::new();
            path.push(start);
            for p in path_polygons {
                path.push(self.polygons[p].centroid);
            }
            path.push(goal);

            Some(path)
        }

        /// Compute a path between two points on the navmesh:
        /// 1. Find the polygons containing the start and goal.
        /// 2. Run A* at the polygon level to get a polygon corridor.
        /// 3. Use the funnel algorithm to refine the path along the polygon edges.
        pub fn find_path(&self, start: Vec3, goal: Vec3) -> Option<Vec<Vec3>> {
            let start_poly = self.find_polygon_for_point(start)?;
            let goal_poly = self.find_polygon_for_point(goal)?;

            // If we're already in the goal polygon, return a trivial path
            if start_poly == goal_poly {
                return Some(vec![start, goal]);
            }

            let poly_path = polygon_astar(&self.polygons, &self.adjacency, start_poly, goal_poly)?;
            let final_path = funnel_path(self, start, goal, &poly_path);
            Some(final_path)
        }
    }

    /// Merges vertices that are closer than `threshold` distance apart.
    /// Updates both `vertices` and `indices` so that duplicate/close vertices are eliminated.
    fn merge_vertices_by_distance(
        vertices: &mut Vec<Vec3>,
        indices: &mut Vec<u32>,
        threshold: f32,
    ) {
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

    /// Extract vertex positions and indices from a mesh, applying the global transform.
    fn extract_mesh_data(mesh: &Mesh, transform: &GlobalTransform) -> (Vec<Vec3>, Vec<u32>) {
        let vertices = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(verts)) => verts,
            _ => panic!("Mesh positions are not Float32x3 format"),
        };

        let indices: Vec<u32> = match mesh.indices().expect("Mesh has no indices") {
            Indices::U32(ind) => ind.to_vec(),
            Indices::U16(ind) => ind.iter().map(|&i| i as u32).collect(),
        };

        let world_vertices: Vec<Vec3> = vertices
            .iter()
            .map(|v| transform.transform_point(Vec3::new(v[0], v[1], v[2])))
            .collect();

        (world_vertices, indices)
    }

    fn build_polygons(vertices: &[Vec3], indices: &[u32]) -> Vec<Polygon> {
        let mut polygons = Vec::new();
        for tri_start in (0..indices.len()).step_by(3) {
            let v1_ix = indices[tri_start];
            let v2_ix = indices[tri_start + 1];
            let v3_ix = indices[tri_start + 2];

            let v1 = vertices[v1_ix as usize];
            let v2 = vertices[v2_ix as usize];
            let v3 = vertices[v3_ix as usize];

            let normal = (v2 - v1).cross(v3 - v1).normalize();
            let centroid = (v1 + v2 + v3) / 3.0;

            polygons.push(Polygon {
                vertex_indices: [v1_ix, v2_ix, v3_ix],
                vertices: [v1, v2, v3],
                normal,
                centroid,
            });
        }
        polygons
    }

    fn build_adjacency(polygons: &[Polygon]) -> Vec<Vec<usize>> {
        let mut edge_map: HashMap<(u32, u32), usize> = HashMap::new();
        let mut adjacency = vec![vec![]; polygons.len()];

        for (i, poly) in polygons.iter().enumerate() {
            // For each polygon, get its edges as pairs of vertex indices
            let edges = polygon_index_edges(poly);
            for edge in edges {
                let sorted_edge = sort_edge_indices(edge);
                if let Some(other_poly) = edge_map.get(&sorted_edge) {
                    adjacency[*other_poly].push(i);
                    adjacency[i].push(*other_poly);
                } else {
                    edge_map.insert(sorted_edge, i);
                }
            }
        }

        adjacency
    }

    fn polygon_index_edges(poly: &Polygon) -> [(u32, u32); 3] {
        [
            (poly.vertex_indices[0], poly.vertex_indices[1]),
            (poly.vertex_indices[1], poly.vertex_indices[2]),
            (poly.vertex_indices[2], poly.vertex_indices[0]),
        ]
    }

    fn sort_edge_indices(edge: (u32, u32)) -> (u32, u32) {
        if edge.0 < edge.1 {
            edge
        } else {
            (edge.1, edge.0)
        }
    }

    /// Run A* on the polygon graph.
    /// Heuristic: Euclidean distance between polygon centroids.
    fn polygon_astar(
        polygons: &Vec<Polygon>,
        adjacency: &Vec<Vec<usize>>,
        start: usize,
        goal: usize,
    ) -> Option<Vec<usize>> {
        // A* data structures
        let n = polygons.len();
        let mut came_from = vec![None; n];
        let mut g_score = vec![f32::INFINITY; n];
        let mut f_score = vec![f32::INFINITY; n];

        g_score[start] = 0.0;
        f_score[start] = polygons[start].centroid.distance(polygons[goal].centroid);

        let mut open_set = BinaryHeap::new();
        let mut in_open_set = HashSet::new();

        open_set.push(NodeCost {
            node: start,
            f_score: f_score[start],
        });
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
                let tentative_g = g_score[current]
                    + polygons[current]
                        .centroid
                        .distance(polygons[neighbor].centroid);
                if tentative_g < g_score[neighbor] {
                    came_from[neighbor] = Some(current);
                    g_score[neighbor] = tentative_g;
                    f_score[neighbor] = tentative_g
                        + polygons[neighbor]
                            .centroid
                            .distance(polygons[goal].centroid);
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

    /// Given two polygon indices, find the shared edge (portal) in world coordinates.
    fn find_shared_edge(poly_a: &Polygon, poly_b: &Polygon) -> Option<(Vec3, Vec3)> {
        // Compare vertex indices to find the common edge
        // An edge is defined by two indices. We look for two matching indices in both polygons.
        let a_ix = &poly_a.vertex_indices;
        let b_ix = &poly_b.vertex_indices;

        let mut shared = Vec::new();
        for &ai in a_ix {
            for &bi in b_ix {
                if ai == bi {
                    shared.push(ai);
                }
            }
        }

        if shared.len() == 2 {
            // The shared edge should appear as two shared indices
            let p1 = poly_a.vertices[a_ix.iter().position(|&x| x == shared[0]).unwrap()];
            let p2 = poly_a.vertices[a_ix.iter().position(|&x| x == shared[1]).unwrap()];
            Some((p1, p2))
        } else {
            None
        }
    }

    #[derive(Copy, Clone, PartialEq)]
    struct NodeCost {
        node: usize,
        f_score: f32,
    }

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

    fn funnel_path(nav: &NavMesh, start: Vec3, goal: Vec3, poly_path: &[usize]) -> Vec<Vec3> {
        if poly_path.is_empty() {
            return vec![start, goal];
        }
    
        // Start by initializing the list of portals
        let mut portals = Vec::new();
        portals.push((start, start)); // Start point as a trivial portal
    
        // Loop through polygon pairs and find shared edges
        for pair in poly_path.windows(2) {
            let a = pair[0];
            let b = pair[1];
    
            if let Some((p1, p2)) = find_shared_edge(&nav.polygons[a], &nav.polygons[b]) {
                // Use the apex as a reference to consistently orient the portal
                let apex = portals.last().unwrap().0;
                if funnel::tri_area2(apex, p1, p2) > 0.0 {
                    portals.push((p1, p2));
                } else {
                    portals.push((p2, p1));
                }
            } else {
                panic!("Failed to find shared edge between polygons {} and {}", a, b);
            }
        }
    
        portals.push((goal, goal)); // End point as a trivial portal
    
        // Run the funnel algorithm on the generated portals
        funnel::compute(&portals)
    }

    mod funnel {
        use bevy::prelude::*;

        pub fn compute(portals: &[(Vec3, Vec3)]) -> Vec<Vec3> {
            let mut path = Vec::new();
            if portals.is_empty() {
                return path;
            }
        
            // Initialize funnel state
            let mut apex = portals[0].0;
            let mut left = portals[1].0;
            let mut right = portals[1].1;
            let mut apex_index = 0;
            let mut left_index = 0;
            let mut right_index = 0;
        
            path.push(apex);
        
            let mut i = 1; // Start processing from the second portal
            while i < portals.len() {
                let (p_left, p_right) = portals[i];
        
                // Update right boundary
                if tri_area2(apex, right, p_right) <= 0.0 {
                    if apex == right || tri_area2(apex, left, p_right) < 0.0 {
                        // Tighten the right boundary
                        right = p_right;
                        right_index = i;
                    } else {
                        // Right crosses left, move apex to left
                        path.push(left);
                        apex = left;
                        apex_index = left_index;
        
                        // Reset funnel
                        left = apex;
                        right = apex;
                        left_index = apex_index;
                        right_index = apex_index;
        
                        i = apex_index + 1; // Restart after the apex, prevent infinite loop
                        continue;
                    }
                }
        
                // Update left boundary
                if tri_area2(apex, left, p_left) >= 0.0 {
                    if apex == left || tri_area2(apex, right, p_left) > 0.0 {
                        // Tighten the left boundary
                        left = p_left;
                        left_index = i;
                    } else {
                        // Left crosses right, move apex to right
                        path.push(right);
                        apex = right;
                        apex_index = right_index;
        
                        // Reset funnel
                        left = apex;
                        right = apex;
                        left_index = apex_index;
                        right_index = apex_index;
        
                        i = apex_index + 1; // Restart after the apex, prevent infinite loop
                        continue;
                    }
                }
        
                // Advance to the next portal
                i += 1;
            }
        
            // Append the last point (goal)
            if *path.last().unwrap() != portals.last().unwrap().0 {
                path.push(portals.last().unwrap().0);
            }
        
            path
        }
        
        /// Twice the signed area of the triangle (a, b, c).
        /// If this is positive, c is to the left of the line ab.
        pub fn tri_area2(a: Vec3, b: Vec3, c: Vec3) -> f32 {
            let ab = b - a;
            let ac = c - a;
            ab.x * ac.z - ab.z * ac.x
        }
    }
}
