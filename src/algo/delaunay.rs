use bevy::prelude::Vec2;
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct Triangle {
    pub vertices: [usize; 3],
}

fn circle_circumcenter(a: Vec2, b: Vec2, c: Vec2) -> Option<(Vec2, f32)> {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if d.abs() < f32::EPSILON {
        return None;
    }
    let a2 = a.length_squared();
    let b2 = b.length_squared();
    let c2 = c.length_squared();
    let ux = (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d;
    let uy = (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d;
    let center = Vec2::new(ux, uy);
    let radius = (a - center).length();
    Some((center, radius))
}

fn point_in_circumcircle(pt: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    if let Some((center, radius)) = circle_circumcenter(a, b, c) {
        return (pt - center).length() < radius;
    }
    false
}

fn supertriangle(points: &[Vec2]) -> [Vec2; 3] {
    let min_x = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
    let max_x = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
    let min_y = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    let max_y = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);

    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let delta_max = dx.max(dy);
    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;

    [
        Vec2::new(mid_x - 2.0 * delta_max, mid_y - delta_max),
        Vec2::new(mid_x, mid_y + 2.0 * delta_max),
        Vec2::new(mid_x + 2.0 * delta_max, mid_y - delta_max),
    ]
}

pub fn delaunay_triangulate(points: &[Vec2]) -> Vec<Triangle> {
    if points.len() < 3 {
        return Vec::new();
    }

    let super_tri = supertriangle(points);
    let mut all_points: Vec<Vec2> = points.to_vec();
    all_points.push(super_tri[0]);
    all_points.push(super_tri[1]);
    all_points.push(super_tri[2]);

    let mut triangles = vec![Triangle {
        vertices: [
            all_points.len() - 3,
            all_points.len() - 2,
            all_points.len() - 1,
        ],
    }];

    for (i, p) in points.iter().enumerate() {
        let mut bad_triangles = Vec::new();
        for (j, triangle) in triangles.iter().enumerate() {
            let a = all_points[triangle.vertices[0]];
            let b = all_points[triangle.vertices[1]];
            let c = all_points[triangle.vertices[2]];

            if point_in_circumcircle(*p, a, b, c) {
                bad_triangles.push(j);
            }
        }

        let mut polygon = HashSet::new();
        for &bt in bad_triangles.iter().rev() {
            let tri = &triangles[bt];
            let edges = [
                (tri.vertices[0], tri.vertices[1]),
                (tri.vertices[1], tri.vertices[2]),
                (tri.vertices[2], tri.vertices[0]),
            ];

            for &edge in &edges {
                if !polygon.remove(&(edge.1, edge.0)) {
                    polygon.insert(edge);
                }
            }

            triangles.swap_remove(bt);
        }

        for &(a, b) in &polygon {
            triangles.push(Triangle {
                vertices: [a, b, i],
            });
        }
    }

    triangles.retain(|triangle| !triangle.vertices.iter().any(|&v| v >= points.len()));

    triangles
}
