use astar::astar;
use avian3d::prelude::*;
use bevy::{color::palettes::css::*, prelude::*, utils::HashSet};

use crate::utils::{self, freecam::FreeCamera};

pub fn run() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PhysicsPlugins::default())
        // .add_plugins(PhysicsDebugPlugin::default())
        .add_plugins(utils::freecam::FreeCameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                setup_navmesh,
                utils::toggle_cursor_grab_with_esc,
                draw_path_nodes,
            ),
        )
        .run();
}

#[derive(Component, Default)]
struct NavGrid {
    data: Option<NavGridData>,
}

struct NavGridData {
    points: Vec<Vec3>,
    adjacency: Vec<Vec<usize>>,
}

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
        NavGrid::default(),
    ));
}

fn setup_navmesh(
    mut commands: Commands,
    mut tracker: Local<HashSet<Entity>>,
    meshes: Query<(Entity, &Mesh3d, &GlobalTransform)>,
    mesh_res: Res<Assets<Mesh>>,
    parents: Query<&Parent>,
    navmesh: Query<&NavGrid>,
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
        let (points, adjacency) = navgrid::generate_nav_points(mesh, transform, 1.0, 1.9);
        commands.entity(navgrid).insert(NavGrid {
            data: Some(NavGridData { points, adjacency }),
        });
        commands
            .entity(entity)
            .insert(ColliderConstructor::TrimeshFromMesh);
    }
}

fn draw_path_nodes(
    mut start: Local<Option<usize>>,
    mut end: Local<Option<usize>>,
    mut path: Local<Option<Vec<usize>>>,
    mut gizmos: Gizmos,
    query: Query<&NavGrid>,
    camera: Query<(&GlobalTransform, &RayCaster, &RayHits), With<Camera>>,
    mouse: Res<ButtonInput<MouseButton>>,
) {
    let Ok((cam_t, ray, hits)) = camera.get_single() else {
        return;
    };
    let Ok(grid) = query.get_single() else {
        return;
    };
    let Some(ref data) = grid.data else {
        return;
    };

    let mut selected: Option<usize> = None;
    if let Some(hit) = hits.iter_sorted().next() {
        let point =
            cam_t.translation() + ray.origin + cam_t.rotation() * *ray.direction * hit.distance;
        gizmos.sphere(Isometry3d::from_translation(point), 0.2, PURPLE);
        for (ix, p) in data.points.iter().enumerate() {
            if p.distance(point) < 0.2 {
                selected = Some(ix);
            }
        }
    }

    if mouse.just_pressed(MouseButton::Left) {
        if start.is_none() || end.is_some() {
            *start = selected;
            *end = None;
        } else if start.is_some() && end.is_none() {
            *end = selected;
        }

        match (*start, *end) {
            (Some(start), Some(end)) => {
                *path = astar(&data.points, &data.adjacency, start, end);
            }
            _ => *path = None,
        }
    }

    for (ix, point) in data.points.iter().enumerate() {
        let isometry = Isometry3d::new(*point, Quat::IDENTITY);
        let color = if selected.is_some_and(|v| v == ix) {
            GREEN
        } else {
            RED
        };
        gizmos.sphere(isometry, 0.1, color);
    }

    if let Some(ref path) = *path {
        for segment in path.windows(2) {
            let start = segment[0];
            let end = segment[1];
            gizmos.line(data.points[start], data.points[end], YELLOW);
        }
    }
}

mod navgrid {
    use bevy::prelude::*;
    use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};

    /// Generate navigation points by projecting a grid onto a transformed mesh.
    ///
    /// # Parameters
    /// - `mesh`: The walkable mesh.
    /// - `mesh_transform`: The global transform of the mesh (applied to its vertices).
    /// - `spacing`: Distance between grid points.
    ///
    /// # Returns
    /// A vector of `(x, y, z)` points on the mesh’s surface.
    pub fn generate_nav_points(
        mesh: &Mesh,
        mesh_transform: &GlobalTransform,
        spacing: f32,
        adjacency_factor: f32,
    ) -> (Vec<Vec3>, Vec<Vec<usize>>) {
        // Compute the AABB (axis-aligned bounding box) in world space
        let (center, half_extents) = compute_aabb_world_space(mesh, mesh_transform);

        // Use the AABB center and half-extents to define the grid bounds
        let grid_min = Vec2::new(center.x - half_extents.x, center.z - half_extents.z);
        let grid_max = Vec2::new(center.x + half_extents.x, center.z + half_extents.z);

        // Retrieve mesh vertex positions
        let vertices = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(verts)) => verts,
            _ => panic!("Mesh positions are not Float32x3 format"),
        };

        let u32_vec: Vec<u32>;
        // Retrieve mesh indices (used for triangle definitions)
        let indices: &[u32] = match mesh.indices().expect("Mesh has no indices") {
            Indices::U32(ind) => ind,
            Indices::U16(ind) => {
                u32_vec = ind.iter().map(|v| *v as u32).collect();
                u32_vec.as_slice()
            }
        };

        let mut nodes = Vec::new();

        // Function to retrieve a vertex and apply the mesh transform
        let get_vertex = |idx: u32| {
            let i = idx as usize;
            let local_pos = Vec3::new(vertices[i][0], vertices[i][1], vertices[i][2]);
            mesh_transform.transform_point(local_pos)
        };

        // Iterate over the grid in the XZ plane
        let mut x_coord = grid_min.x;
        while x_coord <= grid_max.x + f32::EPSILON {
            let mut z_coord = grid_min.y;
            while z_coord <= grid_max.y + f32::EPSILON {
                // Current test point in XZ plane
                let test_point_2d = Vec2::new(x_coord, z_coord);

                // Check if the point intersects any triangle in the mesh
                for tri_start in (0..indices.len()).step_by(3) {
                    // Get the vertices of the current triangle
                    let v1 = get_vertex(indices[tri_start]);
                    let v2 = get_vertex(indices[tri_start + 1]);
                    let v3 = get_vertex(indices[tri_start + 2]);

                    // Check if the XZ point is inside the triangle
                    if is_point_in_triangle_2d(test_point_2d, v1, v2, v3) {
                        // Solve for Y to get the full 3D position
                        if let Some(y) = solve_plane_y(test_point_2d, v1, v2, v3) {
                            nodes.push(Vec3::new(test_point_2d.x, y, test_point_2d.y));
                            break; // Move to the next grid point once we find an intersection
                        }
                    }
                }

                // Move to the next point in the Z direction
                z_coord += spacing;
            }
            // Move to the next point in the X direction
            x_coord += spacing;
        }

        let mut adjacency: Vec<Vec<usize>> = vec![vec![]; nodes.len()];
        for (i, node) in nodes.iter().enumerate() {
            for (j, other) in nodes.iter().enumerate() {
                if i != j && node.distance_squared(*other) < (spacing * adjacency_factor).powf(2.0)
                {
                    adjacency[i].push(j);
                }
            }
        }

        (nodes, adjacency)
    }

    /// Compute the AABB (Axis-Aligned Bounding Box) of a mesh in world space.
    /// Applies the global transform to each vertex to account for scaling, rotation, and translation.
    fn compute_aabb_world_space(mesh: &Mesh, mesh_transform: &GlobalTransform) -> (Vec3, Vec3) {
        // Retrieve vertex positions
        let vertices = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(verts)) => verts,
            _ => panic!("Mesh positions are not Float32x3 format"),
        };

        // Initialize min and max points
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);

        // Transform each vertex to world space and adjust the AABB bounds
        for v in vertices {
            let world_v = mesh_transform.transform_point(Vec3::new(v[0], v[1], v[2]));
            min = min.min(world_v);
            max = max.max(world_v);
        }

        // Compute the center and half-extents of the AABB
        let center = (min + max) * 0.5;
        let half_extents = (max - min) * 0.5;
        (center, half_extents)
    }

    /// Check if a 2D point `(x, z)` lies inside the triangle formed by `v1`, `v2`, and `v3` (in XZ space).
    fn is_point_in_triangle_2d(p: Vec2, v1: Vec3, v2: Vec3, v3: Vec3) -> bool {
        let a = Vec2::new(v1.x, v1.z);
        let b = Vec2::new(v2.x, v2.z);
        let c = Vec2::new(v3.x, v3.z);

        // Compute the total area of the triangle
        let area_orig = triangle_area(a, b, c);

        // Compute sub-areas for each sub-triangle
        let area_1 = triangle_area(p, b, c);
        let area_2 = triangle_area(a, p, c);
        let area_3 = triangle_area(a, b, p);

        // The point is inside if the sum of sub-areas equals the total area (within a small epsilon)
        ((area_1 + area_2 + area_3) - area_orig).abs() < 1e-6
    }

    /// Compute the area of a triangle given by three 2D points.
    fn triangle_area(p1: Vec2, p2: Vec2, p3: Vec2) -> f32 {
        ((p1.x * (p2.y - p3.y)) + (p2.x * (p3.y - p1.y)) + (p3.x * (p1.y - p2.y))).abs() / 2.0
    }

    /// Solve for the Y coordinate of a point in the XZ plane on a triangle's plane.
    fn solve_plane_y(point: Vec2, v1: Vec3, v2: Vec3, v3: Vec3) -> Option<f32> {
        // Compute the plane normal
        let edge1 = v2 - v1;
        let edge2 = v3 - v1;
        let normal = edge1.cross(edge2);

        // If normal.y is near zero, the plane is nearly vertical, and we can't solve for Y
        if normal.y.abs() < f32::EPSILON {
            return None;
        }

        // Compute the plane equation coefficients
        let d = -normal.dot(v1);
        let x = point.x;
        let z = point.y;

        // Solve for Y: b*y = -(a*x + c*z + d)
        let y = -(normal.x * x + normal.z * z + d) / normal.y;
        Some(y)
    }
}

mod astar {
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
}
