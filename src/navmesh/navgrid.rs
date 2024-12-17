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
