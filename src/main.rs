use std::f32::consts::FRAC_PI_2;

use anim::PlayerProceduralAnimationTargets;
use bevy::{animation::AnimationTarget, color::palettes::css::*, prelude::*};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use state::{PlayerAnimationInput, PlayerAnimationState};
use utils::{freecam::FreeCamera, toggle_cursor_grab_with_esc};

mod anim;
mod state;
mod utils;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(WorldInspectorPlugin::new())
        .add_plugins(utils::freecam::FreeCameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                draw_xyz_gizmo,
                init_player_animations,
                transition_player_animations,
                toggle_cursor_grab_with_esc,
                toggle_freecam,
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

        let (anims, proc_targets, graph) = anim::load_player_animations(
            entity,
            &asset_server,
            &children,
            &names,
            &animation_targets,
        );

        commands
            .entity(entity)
            .insert(AnimationGraphHandle(animation_graphs.add(graph)))
            .insert(PlayerAnimationState::new(anims))
            .insert(proc_targets);
    }
}

fn transition_player_animations(
    mut look_x_rotation: Local<f32>,
    mut airborne: Local<bool>,
    keys: Res<ButtonInput<KeyCode>>,
    mut players: Query<(
        &mut AnimationPlayer,
        &mut PlayerAnimationState,
        &PlayerProceduralAnimationTargets,
    )>,
    player_roots: Query<Entity, With<Player>>,
    mut transforms: Query<&mut Transform>,
    global_transforms: Query<&GlobalTransform>,
) {
    let local_movement_direction = utils::unit_vector_from_bools(
        keys.pressed(KeyCode::KeyW),
        keys.pressed(KeyCode::KeyS),
        keys.pressed(KeyCode::KeyA),
        keys.pressed(KeyCode::KeyD),
    );

    let input = PlayerAnimationInput {
        just_jumped: !*airborne && keys.just_pressed(KeyCode::KeyJ),
        is_grounded: !*airborne,
        local_movement_direction,
    };

    if keys.just_pressed(KeyCode::KeyJ) {
        *airborne = !*airborne;
    }

    if keys.pressed(KeyCode::ArrowUp) {
        *look_x_rotation += 1f32.to_radians();
    }
    if keys.pressed(KeyCode::ArrowDown) {
        *look_x_rotation -= 1f32.to_radians();
    }
    let mut body_y_rotation = 0.0;
    if keys.pressed(KeyCode::ArrowLeft) {
        body_y_rotation += 1f32.to_radians();
    }
    if keys.pressed(KeyCode::ArrowRight) {
        body_y_rotation -= 1f32.to_radians();
    }

    let Ok(root_entity) = player_roots.get_single() else {
        return;
    };
    let mut root_local = transforms
        .get_mut(root_entity)
        .expect("root should have transform");
    root_local.rotate_local_y(body_y_rotation);

    for (mut player, mut state, proc_targets) in players.iter_mut() {
        state.transition(&input, &player);
        state.update_player(&mut player);
        let mut spine1_local = transforms
            .get_mut(proc_targets.spine1)
            .expect("spine1 should have transform");

        let bullet_point_global = global_transforms.get(proc_targets.bullet_point).unwrap();
        let root_global = global_transforms.get(root_entity).unwrap();
        rotate_spine_to_x(
            root_global,
            bullet_point_global,
            &mut spine1_local,
            *look_x_rotation,
        );
    }
}

/// Rotates the spine bone to the target rotation about the player x-axis.
fn rotate_spine_to_x(
    player_global: &GlobalTransform,
    bullet_point_global: &GlobalTransform,
    spine1_local: &mut Transform,
    target_gun_x_rotation: f32,
) {
    if target_gun_x_rotation > FRAC_PI_2 * 0.9 || target_gun_x_rotation < -FRAC_PI_2 * 0.9 {
        warn!("gun x rotation too large");
        return;
    }

    // Compute the target gun rotation in bullet space.
    let global_target = player_global.rotation()
        * Quat::from_axis_angle(Vec3::X, target_gun_x_rotation)
        * Vec3::Z;

    // Compute the current forward direction of the bullet_point in global space.
    let current_bullet_forward = bullet_point_global.rotation() * Vec3::Y;

    // Compute the rotation needed to align the current bullet_point forward to the target direction.
    let alignment_rotation = Quat::from_rotation_arc(current_bullet_forward, global_target);

    // Adjust the spine's local rotation to include the alignment.
    spine1_local.rotation = spine1_local.rotation * alignment_rotation;
    // Stop Z rotation so that the player stays upright.
    let (x, y, _) = spine1_local.rotation.to_euler(EulerRot::XYZ);
    spine1_local.rotation = Quat::from_euler(EulerRot::XYZ, x, y, 0.0);
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
