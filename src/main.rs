use std::env;

use bevy::{
    animation::AnimationTarget,
    color::palettes::css::*,
    prelude::*,
    render::{mesh::skinning::SkinnedMesh, view::NoFrustumCulling},
};
use state::{PlayerAnimationInput, PlayerAnimationState};
use tracer::Tracer;
use utils::{freecam::FreeCamera, toggle_cursor_grab_with_esc};

mod anim;
mod state;
mod tracer;
mod utils;
mod mutant;
mod navmesh;
mod enemy;

fn main() {
    if env::args().any(|v| v == "navmesh") {
        navmesh::run();
        return;
    }
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(utils::freecam::FreeCameraPlugin)
        .add_plugins(tracer::TracerPlugin)
        .add_plugins(anim::AnimationPlugin)
        .add_plugins(mutant::MutantPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                draw_xyz_gizmo,
                init_player_animations,
                transition_player_animations,
                toggle_cursor_grab_with_esc,
                toggle_freecam,
                disable_culling_for_skinned_meshes,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
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

    // Spawn the player character.
    commands.spawn((
        SceneRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/gltf/character.glb")),
        ),
        Player,
        Name::new("Player"),
        Transform::from_scale(Vec3::splat(1.0)),
    ));

    // Spawn the ground.
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(7.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
}

fn toggle_freecam(
    mut enabled: Local<bool>,
    mut freecam: Query<&mut FreeCamera>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    if keys.just_pressed(KeyCode::KeyF) {
        *enabled = !*enabled;
    }
    freecam.single_mut().movement_enabled = *enabled;
}

#[derive(Component)]
struct Player;

fn init_player_animations(
    mut commands: Commands,
    mut new_anim_players: Query<Entity, Added<AnimationPlayer>>,
    asset_server: Res<AssetServer>,
    children: Query<&Children>,
    parents: Query<&Parent>,
    names: Query<&Name>,
    players: Query<&Player>,
    mut animation_graphs: ResMut<Assets<AnimationGraph>>,
    animation_targets: Query<&AnimationTarget>,
) {
    for entity in new_anim_players.iter_mut() {
        if !utils::find_upwards(entity, &parents, &players).is_some() {
            // This is not a player.
            continue;
        };

        let (anims, proc_targets, graph, nodes) = anim::load_player_animations(
            entity,
            &asset_server,
            &children,
            &names,
            &animation_targets,
            commands.reborrow(),
            &parents,
        );

        commands
            .entity(entity)
            .insert(AnimationGraphHandle(animation_graphs.add(graph)))
            .insert(PlayerAnimationState::new(anims, proc_targets, nodes));
    }
}

fn transition_player_animations(
    mut look_x_rotation: Local<f32>,
    mut look_y_rotation: Local<f32>,
    mut airborne: Local<bool>,
    keys: Res<ButtonInput<KeyCode>>,
    mut players: Query<&mut PlayerAnimationState>,
    global_transforms: Query<&GlobalTransform>,
    mut commands: Commands,
) {
    let local_movement_direction = utils::unit_vector_from_bools(
        keys.pressed(KeyCode::KeyW),
        keys.pressed(KeyCode::KeyS),
        keys.pressed(KeyCode::KeyA),
        keys.pressed(KeyCode::KeyD),
    );

    if keys.pressed(KeyCode::ArrowUp) {
        *look_x_rotation += 1f32.to_radians();
    }
    if keys.pressed(KeyCode::ArrowDown) {
        *look_x_rotation -= 1f32.to_radians();
    }
    if keys.pressed(KeyCode::ArrowLeft) {
        *look_y_rotation += 1f32.to_radians();
    }
    if keys.pressed(KeyCode::ArrowRight) {
        *look_y_rotation -= 1f32.to_radians();
    }

    let is_grounded = !*airborne;
    let is_sprinting = is_grounded
        && keys.pressed(KeyCode::ShiftLeft)
        && utils::most_aligned(local_movement_direction) == IVec2::Y;
    let input = PlayerAnimationInput {
        just_jumped: !*airborne && keys.just_pressed(KeyCode::KeyJ),
        is_sprinting,
        is_grounded,
        local_movement_direction,
        look_y: *look_y_rotation,
        look_x: *look_x_rotation,
    };

    if keys.just_pressed(KeyCode::KeyJ) {
        *airborne = !*airborne;
    }

    if let Ok(mut state) = players.get_single_mut() {
        state.set_input(input);

        let bullet_point_global = global_transforms
            .get(state.proc_targets.bullet_point)
            .unwrap();
        if keys.just_pressed(KeyCode::KeyT) {
            commands.spawn((
                Tracer {
                    end: bullet_point_global.translation()
                        + bullet_point_global.rotation() * Vec3::Z * 10.0,
                },
                Transform::from_translation(
                    bullet_point_global.translation()
                        + bullet_point_global.rotation() * Vec3::Z * 0.3,
                ),
            ));
        }
    }
}

fn draw_xyz_gizmo(mut gizmos: Gizmos) {
    let origin = Vec3::ZERO;
    let length = 1.0;

    // X Axis (Red)
    gizmos.line(origin, Vec3::new(length, 0.0, 0.0), RED);
    // Y Axis (Green)
    gizmos.line(origin, Vec3::new(0.0, length, 0.0), GREEN);
    // Z Axis (Blue)
    gizmos.line(origin, Vec3::new(0.0, 0.0, length), BLUE);
}

fn disable_culling_for_skinned_meshes(
    mut commands: Commands,
    skinned: Query<Entity, Added<SkinnedMesh>>,
) {
    for entity in &skinned {
        commands.entity(entity).insert(NoFrustumCulling);
    }
}
