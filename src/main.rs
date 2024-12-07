use bevy::{
    animation::{ActiveAnimation, AnimationTarget, RepeatAnimation},
    asset::AssetPath,
    prelude::*,
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use utils::{freecam::FreeCamera, toggle_cursor_grab_with_esc};

mod utils;

const PLAYER_ANIM_INDICES: [&str; 11] = [
    "AimingIdle",
    "FallingIdle",
    "FallToLand",
    "FiringRifle",
    "Jump",
    "RifleWalkBack",
    "RifleWalkForward",
    "StrafeLeft",
    "StrafeRight",
    "WalkBack",
    "WalkForward",
];

fn get_anim(name: &str) -> AssetPath<'static> {
    for (i, curr) in PLAYER_ANIM_INDICES.iter().enumerate() {
        if name == *curr {
            return GltfAssetLabel::Animation(i).from_asset("character.glb");
        }
    }
    panic!("no anim with name {}", name);
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(WorldInspectorPlugin::new())
        .add_plugins(utils::freecam::FreeCameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
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
        SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("character.glb"))),
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

struct PlayerAnimations {
    forward: AssetPath<'static>,
    back: AssetPath<'static>,
    idle_upper: AssetPath<'static>,
    idle_lower: AssetPath<'static>,
    left: AssetPath<'static>,
    right: AssetPath<'static>,
    jump: AssetPath<'static>,
    falling: AssetPath<'static>,
    land: AssetPath<'static>,
}

impl Default for PlayerAnimations {
    fn default() -> Self {
        PlayerAnimations {
            idle_upper: get_anim("AimingIdle"),
            idle_lower: get_anim("AimingIdle"),
            back: get_anim("RifleWalkBack"),
            forward: get_anim("RifleWalkForward"),
            left: get_anim("StrafeLeft"),
            right: get_anim("StrafeRight"),
            jump: get_anim("Jump"),
            falling: get_anim("FallingIdle"),
            land: get_anim("FallToLand"),
        }
    }
}

#[derive(Component)]
struct PlayerAnimationIndicies {
    upper: UpperBodyAnimations,
    lower: LowerBodyAnimations,
}

struct UpperBodyAnimations {
    idle: AnimationNodeIndex,
}

struct LowerBodyAnimations {
    forward: AnimationNodeIndex,
    back: AnimationNodeIndex,
    idle: AnimationNodeIndex,
    left: AnimationNodeIndex,
    right: AnimationNodeIndex,
    jump: AnimationNodeIndex,
    falling: AnimationNodeIndex,
    land: AnimationNodeIndex,
}

impl PlayerAnimationIndicies {
    fn is_upper_body_anim(&self, index: AnimationNodeIndex) -> bool {
        index == self.upper.idle
    }

    fn is_lower_body_anim(&self, index: AnimationNodeIndex) -> bool {
        index == self.lower.idle
            || index == self.lower.forward
            || index == self.lower.back
            || index == self.lower.right
            || index == self.lower.left
            || index == self.lower.jump
            || index == self.lower.falling
            || index == self.lower.land
    }

    fn get_speed(&self, index: AnimationNodeIndex) -> f32 {
        if index == self.lower.back {
            return 0.7;
        } else if index == self.lower.right {
            return 1.5;
        } else if index == self.lower.forward {
            return 1.5;
        };
        return 1.0;
    }

    fn get_repeat(&self, index: AnimationNodeIndex) -> RepeatAnimation {
        if index == self.lower.jump {
            RepeatAnimation::Never
        } else if index == self.lower.land {
            RepeatAnimation::Never
        } else {
            RepeatAnimation::Forever
        }
    }
}

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
        let upper_body_mask_group = 1;
        // Apply this to mask out (not play) upper body part of this clip.
        let upper_body_mask = 1 << upper_body_mask_group;
        let lower_body_mask_group = 2;
        // Apply this to mask out (not play) lower body part of this clip.
        let lower_body_mask = 1 << lower_body_mask_group;

        if !find_upwards(entity, &parents, &players).is_some() {
            // This is not a player.
            continue;
        };

        let player_anims = PlayerAnimations::default();
        let mut graph = AnimationGraph::new();
        let add_node = graph.add_additive_blend(1.0, graph.root);
        let lower_body_blend = graph.add_blend(1.0, add_node);
        let upper_body_blend = graph.add_blend(1.0, add_node);

        let forward_clip = asset_server.load(player_anims.forward.clone());
        let forward =
            graph.add_clip_with_mask(forward_clip, upper_body_mask, 1.0, lower_body_blend);

        let back_clip = asset_server.load(player_anims.back.clone());
        let back = graph.add_clip_with_mask(back_clip, upper_body_mask, 1.0, lower_body_blend);

        let rifle_clip_upper = asset_server.load(player_anims.idle_upper.clone());
        let idle_upper =
            graph.add_clip_with_mask(rifle_clip_upper, lower_body_mask, 1.0, upper_body_blend);

        let rifle_clip_lower = asset_server.load(player_anims.idle_lower.clone());
        let idle_lower =
            graph.add_clip_with_mask(rifle_clip_lower, upper_body_mask, 1.0, lower_body_blend);

        let left_clip = asset_server.load(player_anims.left.clone());
        let left = graph.add_clip_with_mask(left_clip, upper_body_mask, 1.0, lower_body_blend);

        let right_clip = asset_server.load(player_anims.right.clone());
        let right = graph.add_clip_with_mask(right_clip, upper_body_mask, 1.0, lower_body_blend);

        let jump_clip = asset_server.load(player_anims.jump.clone());
        let jump = graph.add_clip_with_mask(jump_clip, upper_body_mask, 1.0, lower_body_blend);

        let falling_clip = asset_server.load(player_anims.falling.clone());
        let falling =
            graph.add_clip_with_mask(falling_clip, upper_body_mask, 1.0, lower_body_blend);

        let land_clip = asset_server.load(player_anims.land.clone());
        let land = graph.add_clip_with_mask(land_clip, upper_body_mask, 1.0, lower_body_blend);

        let proc_targets = init_mixamo_rig_masks(
            entity,
            &mut graph,
            upper_body_mask_group,
            lower_body_mask_group,
            &children,
            &names,
            &animation_targets,
        );

        commands
            .entity(entity)
            .insert(AnimationGraphHandle(animation_graphs.add(graph)))
            .insert(PlayerAnimationState::default())
            .insert(proc_targets)
            .insert(PlayerAnimationIndicies {
                upper: UpperBodyAnimations { idle: idle_upper },
                lower: LowerBodyAnimations {
                    forward,
                    back,
                    idle: idle_lower,
                    left,
                    right,
                    jump,
                    land,
                    falling,
                },
            });
    }
}

#[derive(Component)]
struct PlayerAnimationState {
    lower_body: LowerBodyState,
}

impl Default for PlayerAnimationState {
    fn default() -> Self {
        Self {
            lower_body: LowerBodyState::Idle,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LowerBodyState {
    Idle,
    Forward,
    Back,
    Left,
    Right,
    Jump,
    Falling,
    Land,
}

struct PlayerAnimationInput {
    /// +Y is forward
    local_movement_direction: Vec2,

    just_jumped: bool,
    is_grounded: bool,
}

fn transition_animation_state(
    input: &PlayerAnimationInput,
    anims: &PlayerAnimationIndicies,
    state: &mut PlayerAnimationState,
    player: &mut AnimationPlayer,
) {
    if state.lower_body == LowerBodyState::Land
        && player.animation(anims.lower.land).unwrap().is_finished()
    {
        state.lower_body = LowerBodyState::Idle;
    } else if state.lower_body == LowerBodyState::Jump {
        // Skip land animation if landed before jumped.
        if input.is_grounded {
            state.lower_body = LowerBodyState::Idle;
        } else if player.animation(anims.lower.jump).unwrap().is_finished() {
            state.lower_body = LowerBodyState::Falling;
        }
    } else if state.lower_body == LowerBodyState::Falling {
        if input.is_grounded {
            state.lower_body = LowerBodyState::Land;
        }
    } else if input.just_jumped {
        state.lower_body = LowerBodyState::Jump;
    } else if input.local_movement_direction.length() < 0.1 {
        state.lower_body = LowerBodyState::Idle;
    } else {
        state.lower_body = *sample_cardinal(
            &[
                LowerBodyState::Forward,
                LowerBodyState::Back,
                LowerBodyState::Left,
                LowerBodyState::Right,
            ],
            input.local_movement_direction,
        );
    }

    let target_lower_body_anim = match state.lower_body {
        LowerBodyState::Idle => anims.lower.idle,
        LowerBodyState::Forward => anims.lower.forward,
        LowerBodyState::Back => anims.lower.back,
        LowerBodyState::Right => anims.lower.right,
        LowerBodyState::Left => anims.lower.left,
        LowerBodyState::Jump => anims.lower.jump,
        LowerBodyState::Falling => anims.lower.falling,
        LowerBodyState::Land => anims.lower.land,
    };

    let animations_to_fade = player
        .playing_animations()
        .filter_map(|(ix, _)| {
            if anims.is_lower_body_anim(*ix) && *ix != target_lower_body_anim {
                Some(*ix)
            } else {
                None
            }
        })
        .collect();

    let rate = 0.9;
    let threshold = 0.01;

    fade_out_animations(player, animations_to_fade, rate, threshold);
    fade_in_animation(player, target_lower_body_anim, 1.0 / rate, threshold)
        .set_speed(anims.get_speed(target_lower_body_anim))
        .set_repeat(anims.get_repeat(target_lower_body_anim));

    let target_upper_body_anim = anims.upper.idle;
    player
        .play(target_upper_body_anim)
        .set_repeat(anims.get_repeat(target_upper_body_anim))
        .set_weight(1.0)
        .set_speed(anims.get_speed(anims.upper.idle));
}

fn fade_out_animations(
    player: &mut AnimationPlayer,
    anims: Vec<AnimationNodeIndex>,
    rate: f32,
    threshold: f32,
) {
    for index in anims {
        let anim = player.animation_mut(index).unwrap();
        if anim.weight() < threshold {
            player.stop(index);
        } else {
            anim.set_weight(anim.weight() * rate);
        }
    }
}

fn fade_in_animation(
    player: &mut AnimationPlayer,
    anim: AnimationNodeIndex,
    rate: f32,
    threshold: f32,
) -> &mut ActiveAnimation {
    let is_playing_target_lower_body_anim = player.is_playing_animation(anim);
    let target_anim = player.play(anim);

    if is_playing_target_lower_body_anim {
        let curr_weight = target_anim.weight();
        target_anim.set_weight((curr_weight * rate).min(1.0));
    } else {
        target_anim.set_weight(threshold);
    }

    target_anim
}

fn transition_player_animations(
    mut airborne: Local<bool>,
    keys: Res<ButtonInput<KeyCode>>,
    mut players: Query<(
        &mut AnimationPlayer,
        &mut PlayerAnimationState,
        &PlayerAnimationIndicies,
        &PlayerProceduralAnimationTargets,
    )>,
    player_roots: Query<Entity, With<Player>>,
    mut transforms: Query<(&mut Transform, &GlobalTransform)>,
    global_transforms: Query<&GlobalTransform>,
) {
    let local_movement_direction = unit_vector_from_bools(
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

    let mut look_x_rotation = 0.0;
    if keys.pressed(KeyCode::ArrowUp) {
        look_x_rotation += 1f32.to_radians();
    }
    if keys.pressed(KeyCode::ArrowDown) {
        look_x_rotation -= 1f32.to_radians();
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

    for (mut player, mut state, anims, proc_targets) in players.iter_mut() {
        transition_animation_state(&input, anims, &mut state, &mut player);
        let (mut spine_local, spine_global) = transforms
            .get_mut(proc_targets.spine)
            .expect("spine1 should have transform");
        let root_global = global_transforms.get(root_entity).unwrap();
        rotate_spine_about_player_x(root_global, spine_global, &mut spine_local, look_x_rotation);
    }
}

#[derive(Component)]
struct PlayerProceduralAnimationTargets {
    spine: Entity,
}

/// groups. The param `entity` should be the entity with an AnimationPlayer.
fn init_mixamo_rig_masks(
    entity: Entity,
    graph: &mut AnimationGraph,
    upper_body_mask_group: u32,
    lower_body_mask_group: u32,
    children: &Query<&Children>,
    names: &Query<&Name>,
    animation_targets: &Query<&AnimationTarget>,
) -> PlayerProceduralAnimationTargets {
    // Joints to mask out. All decendants (and this one) will be masked out.
    let upper_body_joint_paths = ["mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1"];
    let lower_body_joint_paths = [
        "mixamorig:Hips/mixamorig:LeftUpLeg",
        "mixamorig:Hips/mixamorig:RightUpLeg",
    ];
    // Isolates don't mask decendants.
    let isolated_lower_body_joint_paths = ["mixamorig:Hips"];
    let isolated_fully_mask = ["mixamorig:Hips/mixamorig:Spine"];

    let mut proc_targets = None;

    for joint_path in isolated_fully_mask {
        let entity = find_child_by_path(entity, joint_path, &children, &names)
            .expect("upper body joint not found");
        let target = animation_targets.get(entity).expect(&format!(
            "isolate {} should have animation target",
            joint_path
        ));
        graph.add_target_to_mask_group(target.id, lower_body_mask_group);
        graph.add_target_to_mask_group(target.id, upper_body_mask_group);

        if joint_path.ends_with("Spine") {
            proc_targets = Some(PlayerProceduralAnimationTargets { spine: entity });
        }
    }

    for joint_path in upper_body_joint_paths {
        // Find entity from joint path.
        let upper_body_joint_entity = find_child_by_path(entity, joint_path, &children, &names)
            .expect("upper body joint not found");

        // Add every joint for every decendant (including the joint path).
        let entities_to_mask = get_all_descendants(upper_body_joint_entity, &children);
        let targets_to_mask = map_query(entities_to_mask, &animation_targets);
        for target in targets_to_mask {
            graph.add_target_to_mask_group(target.id, upper_body_mask_group);
        }
    }

    // Same thing here for both legs.
    for joint_path in lower_body_joint_paths {
        let lower_body_joint_entity = find_child_by_path(entity, joint_path, &children, &names)
            .expect("lower body joint not found");

        let entities_to_mask = get_all_descendants(lower_body_joint_entity, &children);
        let targets_to_mask = map_query(entities_to_mask, &animation_targets);
        for target in targets_to_mask.iter() {
            graph.add_target_to_mask_group(target.id, lower_body_mask_group);
        }
    }

    // The root of the character (mixamorig:Hips) is still animated by both upper and
    // lower. It is bad to have the same target animated twice by an additive node. Here
    // we decide to assign the hip bone (but not decendants, which we already assigned to
    // either upper or lower) to the lower body.
    for isolate in isolated_lower_body_joint_paths {
        let entity = find_child_by_path(entity, isolate, &children, &names)
            .expect(&format!("isolate bone {} should exist", isolate));
        let target = animation_targets
            .get(entity)
            .expect(&format!("isolate {} should have animation target", isolate));
        graph.add_target_to_mask_group(target.id, lower_body_mask_group);
    }

    proc_targets.expect("proc target to be set")
}

/// Recursively searches for a child entity by a path of names, starting from the given root entity.
/// Returns the child entity if found, or `None` if the path is invalid/entity cannot be found.
fn find_child_by_path(
    scene: Entity,
    path: &str,
    children: &Query<&Children>,
    names: &Query<&Name>,
) -> Option<Entity> {
    let mut parent = scene;

    for segment in path.split('/') {
        let old_parent = parent;

        if let Ok(child_entities) = children.get(parent) {
            for &child in child_entities {
                if let Ok(name) = names.get(child) {
                    if name.as_str() == segment {
                        parent = child;
                        break;
                    }
                }
            }
        }

        if old_parent == parent {
            return None;
        }
    }

    Some(parent)
}

/// Gets all decendants recursivley, including `entity`.
fn get_all_descendants(entity: Entity, children: &Query<&Children>) -> Vec<Entity> {
    let Ok(children_ok) = children.get(entity) else {
        return vec![entity];
    };
    children_ok
        .iter()
        .flat_map(|e| get_all_descendants(*e, children))
        .chain(std::iter::once(entity))
        .collect()
}

/// Queries a component for a list of entities.
fn map_query<T: Component + Clone>(entites: Vec<Entity>, query: &Query<&T>) -> Vec<T> {
    entites
        .into_iter()
        .flat_map(|v| query.get(v).ok())
        .cloned()
        .collect::<Vec<_>>()
}

fn find_upwards<'a, T: Component>(
    entity: Entity,
    parents: &Query<&Parent>,
    component: &'a Query<&T>,
) -> Option<&'a T> {
    let mut search = entity;
    while let Ok(parent) = parents.get(search) {
        if let Ok(comp) = component.get(parent.get()) {
            return Some(comp);
        };
        search = parent.get();
    }
    return None;
}

/// Samples from the element in the array corresponding to the most aligned cardinal direction (forward, back, left, right) based on a Vec2.
fn sample_cardinal<T>(array: &[T; 4], direction: Vec2) -> &T {
    // Normalize the direction vector to handle non-unit vectors
    let normalized_dir = direction.normalize_or_zero();
    // Define the cardinal directions as unit vectors
    let cardinal_directions = [
        Vec2::new(0.0, 1.0),  // Forward
        Vec2::new(0.0, -1.0), // Back
        Vec2::new(-1.0, 0.0), // Left
        Vec2::new(1.0, 0.0),  // Right
    ];
    // Find the index of the most aligned cardinal direction
    let (max_index, _) = cardinal_directions
        .iter()
        .enumerate()
        .map(|(i, &cardinal)| (i, normalized_dir.dot(cardinal)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    &array[max_index]
}

/// Constructs a unit vector (or zero vector) based on 4 booleans: [forward, back, left, right].
fn unit_vector_from_bools(forward: bool, back: bool, left: bool, right: bool) -> Vec2 {
    let mut vec = Vec2::ZERO;
    if forward {
        vec += Vec2::new(0.0, 1.0);
    }
    if back {
        vec += Vec2::new(0.0, -1.0);
    }
    if left {
        vec += Vec2::new(-1.0, 0.0);
    }
    if right {
        vec += Vec2::new(1.0, 0.0);
    }
    vec.normalize_or_zero()
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
