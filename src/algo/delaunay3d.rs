use bevy::math::Vec4;
use bevy::prelude::{Mat4, Vec3};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4::new(x, y, z, w)
}

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: Vec3,
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let px = self.position.x.to_bits();
        let py = self.position.y.to_bits();
        let pz = self.position.z.to_bits();
        px.hash(state);
        py.hash(state);
        pz.hash(state);
    }
}

#[inline]
fn almost_equal_vertex(left: &Vertex, right: &Vertex) -> bool {
    (left.position - right.position).length_squared() < 0.01f32
}

#[derive(Clone, Debug)]
pub struct Tetrahedron {
    pub a: Vertex,
    pub b: Vertex,
    pub c: Vertex,
    pub d: Vertex,
    pub is_bad: bool,
    circumcenter: Vec3,
    circumradius_squared: f32,
}

impl Tetrahedron {
    pub fn new(a: Vertex, b: Vertex, c: Vertex, d: Vertex) -> Self {
        let mut t = Tetrahedron {
            a,
            b,
            c,
            d,
            is_bad: false,
            circumcenter: Vec3::ZERO,
            circumradius_squared: 0.0,
        };
        t.calculate_circumsphere();
        t
    }

    fn calculate_circumsphere(&mut self) {
        // Based on the formula from:
        // http://mathworld.wolfram.com/Circumsphere.html

        let a_det = {
            // Construct matrix for a
            let m = Mat4::from_cols(
                vec4(self.a.position.x, self.b.position.x, self.c.position.x, self.d.position.x),
                vec4(self.a.position.y, self.b.position.y, self.c.position.y, self.d.position.y),
                vec4(self.a.position.z, self.b.position.z, self.c.position.z, self.d.position.z),
                vec4(1.0, 1.0, 1.0, 1.0),
            );
            m.determinant()
        };

        let a_pos_sqr = self.a.position.length_squared();
        let b_pos_sqr = self.b.position.length_squared();
        let c_pos_sqr = self.c.position.length_squared();
        let d_pos_sqr = self.d.position.length_squared();

        let dx = {
            let m = Mat4::from_cols(
                vec4(a_pos_sqr, b_pos_sqr, c_pos_sqr, d_pos_sqr),
                vec4(self.a.position.y, self.b.position.y, self.c.position.y, self.d.position.y),
                vec4(self.a.position.z, self.b.position.z, self.c.position.z, self.d.position.z),
                vec4(1.0, 1.0, 1.0, 1.0),
            );
            m.determinant()
        };

        let dy = {
            let m = Mat4::from_cols(
                vec4(a_pos_sqr, b_pos_sqr, c_pos_sqr, d_pos_sqr),
                vec4(self.a.position.x, self.b.position.x, self.c.position.x, self.d.position.x),
                vec4(self.a.position.z, self.b.position.z, self.c.position.z, self.d.position.z),
                vec4(1.0, 1.0, 1.0, 1.0),
            );
            -m.determinant() // Note the negative sign as in the original code
        };

        let dz = {
            let m = Mat4::from_cols(
                vec4(a_pos_sqr, b_pos_sqr, c_pos_sqr, d_pos_sqr),
                vec4(self.a.position.x, self.b.position.x, self.c.position.x, self.d.position.x),
                vec4(self.a.position.y, self.b.position.y, self.c.position.y, self.d.position.y),
                vec4(1.0, 1.0, 1.0, 1.0),
            );
            m.determinant()
        };

        let c_det = {
            let m = Mat4::from_cols(
                vec4(a_pos_sqr, b_pos_sqr, c_pos_sqr, d_pos_sqr),
                vec4(self.a.position.x, self.b.position.x, self.c.position.x, self.d.position.x),
                vec4(self.a.position.y, self.b.position.y, self.c.position.y, self.d.position.y),
                vec4(self.a.position.z, self.b.position.z, self.c.position.z, self.d.position.z),
            );
            m.determinant()
        };

        self.circumcenter = Vec3::new(
            dx / (2.0 * a_det),
            dy / (2.0 * a_det),
            dz / (2.0 * a_det),
        );

        self.circumradius_squared = ((dx * dx) + (dy * dy) + (dz * dz) - (4.0 * a_det * c_det)) / (4.0 * a_det * a_det);
    }

    pub fn contains_vertex(&self, v: &Vertex) -> bool {
        almost_equal_vertex(v, &self.a)
            || almost_equal_vertex(v, &self.b)
            || almost_equal_vertex(v, &self.c)
            || almost_equal_vertex(v, &self.d)
    }

    pub fn circum_circle_contains(&self, v: Vec3) -> bool {
        let dist = v - self.circumcenter;
        dist.length_squared() <= self.circumradius_squared
    }
}

impl PartialEq for Tetrahedron {
    fn eq(&self, other: &Self) -> bool {
        // Matches the original operator==
        (self.a == other.a || self.a == other.b || self.a == other.c || self.a == other.d) &&
        (self.b == other.a || self.b == other.b || self.b == other.c || self.b == other.d) &&
        (self.c == other.a || self.c == other.b || self.c == other.c || self.c == other.d) &&
        (self.d == other.a || self.d == other.b || self.d == other.c || self.d == other.d)
    }
}

impl Eq for Tetrahedron {}

impl Hash for Tetrahedron {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Just XORing is similar to what the original code did. We'll just hash all vertices.
        self.a.hash(state);
        self.b.hash(state);
        self.c.hash(state);
        self.d.hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct Triangle {
    pub u: Vertex,
    pub v: Vertex,
    pub w: Vertex,
    pub is_bad: bool,
}

impl Triangle {
    pub fn new(u: Vertex, v: Vertex, w: Vertex) -> Self {
        Triangle { u, v, w, is_bad: false }
    }

    pub fn almost_equal(left: &Triangle, right: &Triangle) -> bool {
        // Matches the original AlmostEqual logic
        (almost_equal_vertex(&left.u, &right.u) || almost_equal_vertex(&left.u, &right.v) || almost_equal_vertex(&left.u, &right.w))
            && (almost_equal_vertex(&left.v, &right.u) || almost_equal_vertex(&left.v, &right.v) || almost_equal_vertex(&left.v, &right.w))
            && (almost_equal_vertex(&left.w, &right.u) || almost_equal_vertex(&left.w, &right.v) || almost_equal_vertex(&left.w, &right.w))
    }
}

impl PartialEq for Triangle {
    fn eq(&self, other: &Self) -> bool {
        (self.u == other.u || self.u == other.v || self.u == other.w) &&
        (self.v == other.u || self.v == other.v || self.v == other.w) &&
        (self.w == other.u || self.w == other.v || self.w == other.w)
    }
}

impl Eq for Triangle {}

impl Hash for Triangle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.u.hash(state);
        self.v.hash(state);
        self.w.hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub u: Vertex,
    pub v: Vertex,
    pub is_bad: bool,
}

impl Edge {
    pub fn new(u: Vertex, v: Vertex) -> Self {
        Edge { u, v, is_bad: false }
    }

    pub fn almost_equal(left: &Edge, right: &Edge) -> bool {
        (almost_equal_vertex(&left.u, &right.u) || almost_equal_vertex(&left.v, &right.u))
            && (almost_equal_vertex(&left.u, &right.v) || almost_equal_vertex(&left.v, &right.v))
    }
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        (self.u == other.u || self.u == other.v) &&
        (self.v == other.u || self.v == other.v)
    }
}

impl Eq for Edge {}

impl Hash for Edge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.u.hash(state);
        self.v.hash(state);
    }
}

pub struct Delaunay3D {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub triangles: Vec<Triangle>,
    pub tetrahedra: Vec<Tetrahedron>,
}

impl Delaunay3D {
    pub fn new() -> Self {
        Delaunay3D {
            vertices: Vec::new(),
            edges: Vec::new(),
            triangles: Vec::new(),
            tetrahedra: Vec::new(),
        }
    }

    pub fn triangulate(vertices: &[Vertex]) -> Delaunay3D {
        let mut delaunay = Delaunay3D::new();
        delaunay.vertices = vertices.to_vec();
        delaunay.do_triangulate();
        delaunay
    }

    fn do_triangulate(&mut self) {
        if self.vertices.is_empty() {
            return;
        }

        let mut min_x = self.vertices[0].position.x;
        let mut min_y = self.vertices[0].position.y;
        let mut min_z = self.vertices[0].position.z;
        let mut max_x = min_x;
        let mut max_y = min_y;
        let mut max_z = min_z;

        for vertex in &self.vertices {
            let p = vertex.position;
            if p.x < min_x { min_x = p.x; }
            if p.x > max_x { max_x = p.x; }
            if p.y < min_y { min_y = p.y; }
            if p.y > max_y { max_y = p.y; }
            if p.z < min_z { min_z = p.z; }
            if p.z > max_z { max_z = p.z; }
        }

        let dx = max_x - min_x;
        let dy = max_y - min_y;
        let dz = max_z - min_z;
        let delta_max = f32::max(dx, f32::max(dy, dz)) * 2.0;

        let p1 = Vertex { position: Vec3::new(min_x - 1.0, min_y - 1.0, min_z - 1.0) };
        let p2 = Vertex { position: Vec3::new(max_x + delta_max, min_y - 1.0, min_z - 1.0) };
        let p3 = Vertex { position: Vec3::new(min_x - 1.0, max_y + delta_max, min_z - 1.0) };
        let p4 = Vertex { position: Vec3::new(min_x - 1.0, min_y - 1.0, max_z + delta_max) };

        self.tetrahedra.push(Tetrahedron::new(p1, p2, p3, p4));

        let vertex_count = self.vertices.len();
        for i in 0..vertex_count {
            let vertex = self.vertices[i];
            let mut triangles = Vec::<Triangle>::new();

            for t in &mut self.tetrahedra {
                if t.circum_circle_contains(vertex.position) {
                    t.is_bad = true;
                    triangles.push(Triangle::new(t.a, t.b, t.c));
                    triangles.push(Triangle::new(t.a, t.b, t.d));
                    triangles.push(Triangle::new(t.a, t.c, t.d));
                    triangles.push(Triangle::new(t.b, t.c, t.d));
                }
            }

            for i in 0..triangles.len() {
                for j in (i+1)..triangles.len() {
                    if Triangle::almost_equal(&triangles[i], &triangles[j]) {
                        triangles[i].is_bad = true;
                        triangles[j].is_bad = true;
                    }
                }
            }

            self.tetrahedra.retain(|t| !t.is_bad);
            triangles.retain(|t| !t.is_bad);

            for triangle in triangles {
                self.tetrahedra.push(Tetrahedron::new(triangle.u, triangle.v, triangle.w, vertex));
            }
        }

        self.tetrahedra.retain(|t| {
            !t.contains_vertex(&p1) && !t.contains_vertex(&p2) && !t.contains_vertex(&p3) && !t.contains_vertex(&p4)
        });

        let mut triangle_set = HashSet::<Triangle>::new();
        let mut edge_set = HashSet::<Edge>::new();

        for t in &self.tetrahedra {
            let abc = Triangle::new(t.a, t.b, t.c);
            let abd = Triangle::new(t.a, t.b, t.d);
            let acd = Triangle::new(t.a, t.c, t.d);
            let bcd = Triangle::new(t.b, t.c, t.d);

            if triangle_set.insert(abc.clone()) {
                self.triangles.push(abc);
            }
            if triangle_set.insert(abd.clone()) {
                self.triangles.push(abd);
            }
            if triangle_set.insert(acd.clone()) {
                self.triangles.push(acd);
            }
            if triangle_set.insert(bcd.clone()) {
                self.triangles.push(bcd);
            }

            let ab = Edge::new(t.a, t.b);
            let bc = Edge::new(t.b, t.c);
            let ca = Edge::new(t.c, t.a);
            let da = Edge::new(t.d, t.a);
            let db = Edge::new(t.d, t.b);
            let dc = Edge::new(t.d, t.c);

            if edge_set.insert(ab.clone()) {
                self.edges.push(ab);
            }
            if edge_set.insert(bc.clone()) {
                self.edges.push(bc);
            }
            if edge_set.insert(ca.clone()) {
                self.edges.push(ca);
            }
            if edge_set.insert(da.clone()) {
                self.edges.push(da);
            }
            if edge_set.insert(db.clone()) {
                self.edges.push(db);
            }
            if edge_set.insert(dc.clone()) {
                self.edges.push(dc);
            }
        }
    }
}
