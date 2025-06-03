use bevy::{color::palettes::css::*, prelude::*};
use bevy_rapier3d::prelude::*;
use nav::{setup_navmesh, NavMesh, NavMeshConstructor};

use crate::{
    enemy::{Enemy, EnemyPath, EnemyPlugin, EnemyState},
    utils::{self, freecam::FreeCamera},
};

mod astar;
mod merge;
mod nav;

pub fn run() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(utils::freecam::FreeCameraPlugin)
        .add_plugins(EnemyPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, setup_navmesh)
        .add_systems(Update, utils::toggle_cursor_grab_with_esc)
        .add_systems(Update, draw_path_nodes)
        .run();
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

fn draw_path_nodes(
    mut enemy: Local<Option<Entity>>,
    mut end: Local<Option<Vec3>>,
    mut path: Local<Option<Vec<Vec3>>>,
    mut commands: Commands,
    rapier: ReadRapierContext,
    mut gizmos: Gizmos,
    query: Query<&NavMesh>,
    camera: Query<&GlobalTransform, With<Camera>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut enemies: Query<(&GlobalTransform, &mut Enemy)>,
) {
    let Ok(cam_t) = camera.single() else {
        return;
    };
    let Ok(grid) = query.single() else {
        return;
    };

    let enemy = enemy.get_or_insert_with(|| commands.spawn(Enemy::new(Vec3::Y * 1.0)).id());

    let ray_origin = cam_t.translation();
    let ray_direction = cam_t.rotation() * Vec3::NEG_Z;
    let context = rapier.single().unwrap();

    let hit = context.cast_ray(
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
            }
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
