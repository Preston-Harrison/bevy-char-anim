use anim::PlayerProceduralAnimationTargets;
use bevy::{
    animation::AnimationTarget,
    color::palettes::css::*,
    prelude::*,
};
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
    mut transforms: Query<(&mut Transform, &GlobalTransform)>,
    global_transforms: Query<&GlobalTransform>,
    mut gizmos: Gizmos,
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
    let (mut root_local, _) = transforms
        .get_mut(root_entity)
        .expect("root should have transform");
    root_local.rotate_local_y(body_y_rotation);

    for (mut player, mut state, proc_targets) in players.iter_mut() {
        state.transition(&input, &player);
        state.update_player(&mut player);
        let (mut spine_local, spine_global) = transforms
            .get_mut(proc_targets.spine1)
            .expect("spine1 should have transform");
        let root_global = global_transforms.get(root_entity).unwrap();
        let gun_global = global_transforms.get(proc_targets.gun).unwrap();
        rotate_spine_for_gun(
            root_global,
            gun_global,
            spine_global,
            &mut spine_local,
            *look_x_rotation,
            &mut gizmos,
        );
    }
}

/// Rotates the spine by a given angle around the player's local X axis.
/// This ensures the rotation is applied relative to the player's orientation.
///
/// # Arguments
/// * `player_global` - The global transform of the player, whose local X axis defines the rotation axis.
/// * `spine_global` - The current global transform of the spine.
/// * `spine_local` - The mutable local transform of the spine bone to be rotated.
/// * `angle_radians` - The angle in radians to rotate the spine by around the player's X axis.
pub fn rotate_spine_about_player_x(
    player_global: &GlobalTransform,
    spine_global: &GlobalTransform,
    spine_local: &mut Transform,
    angle_radians: f32,
) {
    let player_x = player_global.rotation() * Vec3::X;
    let spine_local_axis = spine_global.rotation().inverse() * player_x;
    spine_local.rotate_local_axis(Dir3::new(spine_local_axis).unwrap(), angle_radians);
}

fn rotate_spine_for_gun(
    player_global: &GlobalTransform,
    gun_global: &GlobalTransform,
    spine_global: &GlobalTransform,
    spine_local: &mut Transform,
    target_gun_x_rotation: f32,
    gizmos: &mut Gizmos,
) {
    let gun_forward_g = gun_global.rotation() * Vec3::Y;
    // Show gun direction in global space.
    gizmos.line(
        gun_global.translation(),
        gun_global.translation() + gun_forward_g * 1.0,
        BLUE,
    );

    let player_z_g = player_global.rotation() * Vec3::Z;
    let player_x_g = player_global.rotation() * Vec3::X;
    let target_rotation = Quat::from_axis_angle(player_x_g, target_gun_x_rotation);
    let look_direction_g = target_rotation * player_z_g;

    // Show look direction in global space.
    gizmos.line(
        player_global.translation(),
        player_global.translation() + look_direction_g * 1.0,
        LIGHT_CYAN,
    );

    let spine_forward_g = spine_global.rotation() * Vec3::Z;
    gizmos.line(
        spine_global.translation(),
        spine_global.translation() + spine_forward_g * 1.0,
        LIGHT_CORAL,
    );
    gizmos.line(
        spine_global.translation(),
        spine_global.translation() + spine_global.rotation() * Vec3::Y * 1.0,
        LIGHT_BLUE,
    );


    let gun_rotation_diff = quat_from_vectors(gun_forward_g, look_direction_g);
    let ideal_gun_rotation = gun_rotation_diff * gun_forward_g;
    gizmos.line(
        gun_global.translation(),
        gun_global.translation() + ideal_gun_rotation * 1.0,
        LIGHT_CORAL,
    );

}

fn quat_from_vectors(a: Vec3, b: Vec3) -> Quat {
    let a_normalized = a.normalize();
    let b_normalized = b.normalize();

    let dot = a_normalized.dot(b_normalized);
    let cross = a_normalized.cross(b_normalized);

    // Handle edge cases
    if dot > 0.9999 {
        // Vectors are nearly identical, no rotation needed
        return Quat::IDENTITY;
    } else if dot < -0.9999 {
        // Vectors are opposite; find an orthogonal vector for a 180° rotation
        let orthogonal = if a_normalized.abs_diff_eq(Vec3::X, 1e-6) {
            Vec3::Y
        } else {
            Vec3::X
        };
        let axis = a_normalized.cross(orthogonal).normalize();
        return Quat::from_axis_angle(axis, std::f32::consts::PI);
    }

    // General case
    let angle = dot.acos();
    let axis = cross.normalize();
    Quat::from_axis_angle(axis, angle)
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
