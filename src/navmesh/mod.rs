use astar::astar;
use bevy::{color::palettes::css::*, prelude::*, utils::HashSet};
use bevy_rapier3d::prelude::*;
use navmesh::{setup_navmesh, NavMesh, NavMeshConstructor};

use crate::{
    enemy::{Enemy, EnemyPath, EnemyPlugin, EnemyState},
    utils::{self, freecam::FreeCamera},
};

mod astar;
mod navgrid;
mod navmesh;

pub fn run() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(utils::freecam::FreeCameraPlugin)
        .add_plugins(EnemyPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                setup_navmesh,
                setup_navgrid,
                utils::toggle_cursor_grab_with_esc,
                draw_path_nodes,
            ),
        )
        .run();
}

#[derive(Component)]
struct NavGridConstructor;

#[derive(Component)]
struct NavGrid {
    points: Vec<Vec3>,
    adjacency: Vec<Vec<usize>>,
}

impl NavGrid {
    pub fn get_closest_point(&self, point: Vec3) -> Option<usize> {
        let mut min_dist = f32::INFINITY;
        let mut closest = None;
        for (i, p) in self.points.iter().enumerate() {
            let dist = point.distance(*p);
            if dist < min_dist {
                min_dist = dist;
                closest = Some(i);
            }
        }
        closest
    }

    pub fn get_path(&self, start: Vec3, end: Vec3) -> Option<Vec<Vec3>> {
        let start = self.get_closest_point(start)?;
        let end = self.get_closest_point(end)?;
        let path = astar(&self.points, &self.adjacency, start, end)?;
        let path_of_points: Vec<Vec3> = path.into_iter().map(|ix| self.points[ix]).collect();
        Some(path_of_points)
    }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Spawn the camera.
    commands.spawn((
        Camera3d::default(),
        FreeCamera::new(4.0),
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

fn setup_navgrid(
    mut commands: Commands,
    mut tracker: Local<HashSet<Entity>>,
    meshes: Query<(Entity, &Mesh3d, &GlobalTransform)>,
    mesh_res: Res<Assets<Mesh>>,
    parents: Query<&Parent>,
    navmesh: Query<&NavGridConstructor>,
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
        commands
            .entity(navgrid)
            .insert(NavGrid { points, adjacency })
            .remove::<NavGridConstructor>();
        commands
            .entity(entity)
            .insert(Collider::from_bevy_mesh(mesh, &ComputedColliderShape::default()).unwrap());
    }
}

fn draw_path_nodes(
    mut enemy: Local<Option<Entity>>,
    mut end: Local<Option<Vec3>>,
    mut path: Local<Option<Vec<Vec3>>>,
    mut commands: Commands,
    rapier: ReadDefaultRapierContext,
    mut gizmos: Gizmos,
    query: Query<&NavMesh>,
    camera: Query<&GlobalTransform, With<Camera>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut enemies: Query<(&GlobalTransform, &mut Enemy)>,
) {
    let Ok(cam_t) = camera.get_single() else {
        return;
    };
    let Ok(grid) = query.get_single() else {
        return;
    };

    let enemy = enemy.get_or_insert_with(|| {
        commands.spawn(Enemy::new(Vec3::Y * 1.0)).id()
    });

    let ray_origin = cam_t.translation();
    let ray_direction = cam_t.rotation() * Vec3::NEG_Z;
    let hit = rapier.cast_ray(
        ray_origin,
        ray_direction,
        100.0,
        false,
        QueryFilter::default(),
    );

    let mut selected: Option<Vec3> = None;
    if let Some(hit) = hit {
        let point = ray_origin + ray_direction * hit.1;
        gizmos.sphere(Isometry3d::from_translation(point), 0.2, PURPLE);
        selected = Some(point);
    }

    if mouse.just_pressed(MouseButton::Left) {
        *end = selected;

        *path = match *end {
            Some(end) => {
                let (transform, mut enemy) = enemies.get_mut(*enemy).unwrap();
                let path = grid.find_path(transform.translation(), end);
                if let Some(ref p) = path {
                    enemy.state = EnemyState::Moving(EnemyPath {
                        path: p.clone(),
                        target_index: 0,
                        speed: 5.0,
                    });
                };
                path
            },
            _ => None,
        };
    }

    // for (ix, point) in grid.points.iter().enumerate() {
    //     let isometry = Isometry3d::new(*point, Quat::IDENTITY);
    //     let color = if selected.is_some_and(|v| v.distance(grid.points[ix]) < 0.1) {
    //         ORANGE
    //     } else if end.is_some_and(|v| v.distance(grid.points[ix]) < 0.1) {
    //         GREEN
    //     } else {
    //         BLACK
    //     };
    //     gizmos.sphere(isometry, 0.1, color);
    // }

    if let Some(ref path) = *path {
        for segment in path.windows(2) {
            let start = segment[0];
            let end = segment[1];
            gizmos.line(start, end, YELLOW);
        }
    }
}