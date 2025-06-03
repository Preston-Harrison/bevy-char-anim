use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
use bevy::prelude::*;
use bevy::platform::collections::HashSet;
use bevy_rapier3d::prelude::*;

use crate::utils;
use super::{astar, merge};

#[derive(Component)]
pub struct NavMeshConstructor;

pub fn setup_navmesh(
    mut commands: Commands,
    mut tracker: Local<HashSet<Entity>>,
    meshes: Query<(Entity, &Mesh3d, &GlobalTransform)>,
    mesh_res: Res<Assets<Mesh>>,
    parents: Query<&ChildOf>,
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
}

impl NavMesh {
    /// Construct a NavMesh from a given Bevy Mesh and its GlobalTransform.
    /// Assumes the mesh is composed of triangles.
    pub fn from_mesh(mesh: &Mesh, transform: &GlobalTransform) -> Self {
        let (mut vertex_positions, mut triangles) = extract_mesh_data(mesh, transform);
        merge::merge_vertices_by_distance(&mut vertex_positions, &mut triangles, 0.1);
        let adjacency = build_adjacency(&triangles, vertex_positions.len());

        NavMesh {
            triangles,
            vertex_positions,
            adjacency,
        }
    }

    pub fn find_path(&self, from: Vec3, to: Vec3) -> Option<Vec<Vec3>> {
        self.find_vertex_path(from, to)
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

        let vertex_path = astar::vertex_astar(&adjacency, &vertex_positions, start_ix, goal_ix)?;
        let world_path: Vec<_> = vertex_path
            .into_iter()
            .map(|v| vertex_positions[v])
            .collect();
        let pulled = string_pull(
            &world_path,
            &self.triangles,
            &self.vertex_positions,
            1.0,
            0.1,
        );

        Some(pulled)
    }
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

/// Simplifies a path of Vec3 positions using the string-pulling algorithm.
/// Skips redundant points while ensuring the path stays within the navmesh.
///
/// # Arguments:
/// * path - A slice of Vec3 world positions representing the original path.
/// * threshold - The vertical distance tolerance for staying within the navmesh.
///
/// # Returns:
/// A simplified path as a vector of Vec3 positions.
pub fn string_pull(
    path: &[Vec3],
    triangles: &[[usize; 3]],
    vertex_positions: &[Vec3],
    threshold: f32,
    step_size: f32,
) -> Vec<Vec3> {
    let mut simplified_path = Vec::new();

    if path.len() <= 2 {
        return path.to_vec();
    }

    simplified_path.push(path[0]);

    for segments in path[1..].windows(2) {
        let curr = segments[0];
        let next = segments[1];
        let last_anchor = simplified_path[simplified_path.len() - 1];

        if !is_visible(
            last_anchor,
            next,
            triangles,
            vertex_positions,
            threshold,
            step_size,
        ) {
            simplified_path.push(curr);
        }
    }

    // Ensure the final point is included
    let goal = *path.last().unwrap();
    simplified_path.push(goal);

    simplified_path
}

fn is_visible(
    from: Vec3,
    to: Vec3,
    triangles: &[[usize; 3]],
    vertex_positions: &[Vec3],
    threshold: f32,
    step_size: f32,
) -> bool {
    let from_xz = from.xz();
    let to_xz = to.xz();
    let mut current = from;
    let mut step = 0;

    while current.distance(to) > 0.1 {
        let dir = to_xz - from_xz;
        let dir_step = step as f32 * step_size * dir;
        let next_xz = from_xz + dir_step;
        let mut next = None;

        for intersect in vertical_intersections(
            triangles,
            vertex_positions,
            Vec3::new(next_xz.x, 0.0, next_xz.y),
        ) {
            if (intersect.y - current.y).abs() < threshold {
                if next.is_some() {
                    warn!("multiple vertical intersections");
                }
                next = Some(intersect);
            }
        }

        let Some(next) = next else {
            return false;
        };

        current = next;
        step += 1;
    }

    true
}

pub fn vertical_intersections(
    triangles: &[[usize; 3]],
    vertex_positions: &[Vec3],
    point: Vec3,
) -> Vec<Vec3> {
    let mut intersections = Vec::new();

    for triangle in triangles {
        // Get the vertices of the triangle
        let v0 = vertex_positions[triangle[0]];
        let v1 = vertex_positions[triangle[1]];
        let v2 = vertex_positions[triangle[2]];

        // Check if the point is inside the triangle in the XZ-plane
        if is_point_in_triangle_xz(point, v0, v1, v2) {
            // Compute the intersection point on the triangle's plane
            if let Some(intersection) = project_to_plane(point, v0, v1, v2) {
                intersections.push(intersection);
            }
        }
    }

    intersections
}

/// Check if a point lies within a triangle in the XZ-plane.
fn is_point_in_triangle_xz(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    // Flatten points to XZ-plane
    let p2d = (p.x, p.z);
    let a2d = (a.x, a.z);
    let b2d = (b.x, b.z);
    let c2d = (c.x, c.z);

    // Compute barycentric coordinates
    let v0 = (b2d.0 - a2d.0, b2d.1 - a2d.1);
    let v1 = (c2d.0 - a2d.0, c2d.1 - a2d.1);
    let v2 = (p2d.0 - a2d.0, p2d.1 - a2d.1);

    let dot00 = v0.0 * v0.0 + v0.1 * v0.1;
    let dot01 = v0.0 * v1.0 + v0.1 * v1.1;
    let dot02 = v0.0 * v2.0 + v0.1 * v2.1;
    let dot11 = v1.0 * v1.0 + v1.1 * v1.1;
    let dot12 = v1.0 * v2.0 + v1.1 * v2.1;

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() < f32::EPSILON {
        return false; // Degenerate triangle
    }

    let inv_denom = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    (u >= 0.0) && (v >= 0.0) && (u + v <= 1.0)
}

/// Compute the vertical projection of a point onto a triangle's plane.
fn project_to_plane(p: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<Vec3> {
    // Compute the triangle's normal
    let normal = (v1 - v0).cross(v2 - v0);
    if normal.length_squared() < f32::EPSILON {
        return None; // Degenerate triangle
    }

    let normal = normal.normalize();

    // Plane equation: N . (P - V0) = 0
    let d = normal.dot(v0);
    let t = (d - normal.dot(p)) / normal.dot(Vec3::Y);

    // Compute the intersection point
    let intersection = p + Vec3::Y * t;

    Some(intersection)
}
