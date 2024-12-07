use bevy::{animation::AnimationTarget, asset::AssetPath, prelude::*};

const PLAYER_ANIM_INDICES: [&str; 8] = [
    "AimingIdle",
    "FiringRifle",
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
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (init_player_animations, transition_player_animations),
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
        Transform::from_scale(Vec3::splat(1.0)),
        PlayerAnimations {
            idle_upper: get_anim("AimingIdle"),
            idle_lower: get_anim("AimingIdle"),
            back: get_anim("RifleWalkBack"),
            forward: get_anim("RifleWalkForward"),
            left: get_anim("StrafeLeft"),
            right: get_anim("StrafeRight"),
        },
    ));

    // Spawn the ground.
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(7.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
}

#[derive(Component)]
struct PlayerAnimations {
    forward: AssetPath<'static>,
    back: AssetPath<'static>,
    idle_upper: AssetPath<'static>,
    idle_lower: AssetPath<'static>,
    left: AssetPath<'static>,
    right: AssetPath<'static>,
}

#[derive(Component)]
struct PlayerAnimationIndicies {
    forward: AnimationNodeIndex,
    back: AnimationNodeIndex,
    idle_upper: AnimationNodeIndex,
    idle_lower: AnimationNodeIndex,
    left: AnimationNodeIndex,
    right: AnimationNodeIndex,
}

fn init_player_animations(
    mut commands: Commands,
    mut new_anim_players: Query<Entity, Added<AnimationPlayer>>,
    asset_server: Res<AssetServer>,
    children: Query<&Children>,
    parents: Query<&Parent>,
    names: Query<&Name>,
    player_anims: Query<&PlayerAnimations>,
    mut animation_graphs: ResMut<Assets<AnimationGraph>>,
    animation_targets: Query<&AnimationTarget>,
) {
    for entity in new_anim_players.iter_mut() {
        let upper_body_mask_group = 1;
        let upper_body_mask = 1 << upper_body_mask_group;
        let lower_body_mask_group = 2;
        let lower_body_mask = 1 << lower_body_mask_group;

        let Some(player_anims) = find_upwards(entity, &parents, &player_anims) else {
            // This is not a player.
            continue;
        };

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
        let rifle_idle_upper =
            graph.add_clip_with_mask(rifle_clip_upper, lower_body_mask, 1.0, upper_body_blend);

        let rifle_clip_lower = asset_server.load(player_anims.idle_lower.clone());
        let rifle_idle_lower =
            graph.add_clip_with_mask(rifle_clip_lower, upper_body_mask, 1.0, lower_body_blend);

        let left_clip = asset_server.load(player_anims.left.clone());
        let left = graph.add_clip_with_mask(left_clip, upper_body_mask, 1.0, lower_body_blend);

        let right_clip = asset_server.load(player_anims.right.clone());
        let right = graph.add_clip_with_mask(right_clip, upper_body_mask, 1.0, lower_body_blend);

        split_mixamo_rig(
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
            .insert(PlayerAnimationIndicies {
                forward,
                back,
                idle_upper: rifle_idle_upper,
                idle_lower: rifle_idle_lower,
                left,
                right,
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
    Airborne,
    Land,
}

struct PlayerAnimationInput {
    /// +Y is forward
    local_movement_direction: Vec2,
    just_jumped: bool,
    is_grounded: bool,
    just_landed: bool,
}

fn transition_animation_state(
    input: &PlayerAnimationInput,
    anims: &PlayerAnimationIndicies,
    state: &mut PlayerAnimationState,
    player: &mut AnimationPlayer,
) {
    if input.local_movement_direction.length() < 0.1 {
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

    let desired_animation = match state.lower_body {
        LowerBodyState::Idle => anims.idle_lower,
        LowerBodyState::Forward => anims.forward,
        LowerBodyState::Back => anims.back,
        LowerBodyState::Right => anims.right,
        LowerBodyState::Left => anims.left,
        _ => todo!(),
    };

    let mut animations_to_stop = vec![];
    for active in player.playing_animations() {
        if *active.0 != desired_animation && *active.0 != anims.idle_upper {
            animations_to_stop.push(*active.0);
        }
    }

    let lerp_speed = 0.9;
    let lerp_threshold = 0.01;

    for i in animations_to_stop {
        let anim = player.animation_mut(i).unwrap();
        if anim.weight() < lerp_threshold {
            player.stop(i);
        } else {
            anim.set_weight(anim.weight() * lerp_speed);
        }
    }

    let is_playing = player.is_playing_animation(desired_animation);
    let desired_anim_mut = player.play(desired_animation).repeat();

    if desired_animation == anims.back {
        desired_anim_mut.set_speed(0.7);
    } else if desired_animation == anims.right {
        desired_anim_mut.set_speed(1.5);
    } else if desired_animation == anims.forward {
        desired_anim_mut.set_speed(1.5);
    }

    if is_playing {
        desired_anim_mut.set_weight((desired_anim_mut.weight() * (1.0 / lerp_speed)).min(1.0));
    } else {
        desired_anim_mut.set_weight(lerp_threshold);
    }
    player.play(anims.idle_upper).repeat().set_weight(1.0);
}

fn transition_player_animations(
    keys: Res<ButtonInput<KeyCode>>,
    mut players: Query<(
        &mut AnimationPlayer,
        &mut PlayerAnimationState,
        &PlayerAnimationIndicies,
    )>,
) {
    let local_movement_direction = unit_vector_from_bools(
        keys.pressed(KeyCode::KeyW),
        keys.pressed(KeyCode::KeyS),
        keys.pressed(KeyCode::KeyA),
        keys.pressed(KeyCode::KeyD),
    );
    let input = PlayerAnimationInput {
        local_movement_direction,
        is_grounded: true,
        just_jumped: false,
        just_landed: false,
    };

    for (mut player, mut state, anims) in players.iter_mut() {
        transition_animation_state(&input, anims, &mut state, &mut player);
    }
}

/// groups. The param `entity` should be the entity with an AnimationPlayer.
fn split_mixamo_rig(
    entity: Entity,
    graph: &mut AnimationGraph,
    upper_body_mask_group: u32,
    lower_body_mask_group: u32,
    children: &Query<&Children>,
    names: &Query<&Name>,
    animation_targets: &Query<&AnimationTarget>,
) {
    // Joints to mask out. All decendants (and this one) will be masked out.
    let upper_body_joint_path = "mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2";
    let lower_body_joint_paths = [
        "mixamorig:Hips/mixamorig:LeftUpLeg",
        "mixamorig:Hips/mixamorig:RightUpLeg",
    ];

    // These will be assigned to lower body, but decendants will not be.
    let isolated_lower_body_joint_paths = [
        "mixamorig:Hips",
        "mixamorig:Hips/mixamorig:Spine",
        "mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1",
    ];

    // Find entity from joint path.
    let upper_body_joint_entity =
        find_child_by_path(entity, upper_body_joint_path, &children, &names)
            .expect("upper body joint not found");

    // Add every joint for every decendant (including the joint path).
    let entities_to_mask = get_all_descendants(upper_body_joint_entity, &children);
    let targets_to_mask = map_query(entities_to_mask, &animation_targets);
    for target in targets_to_mask {
        graph.add_target_to_mask_group(target.id, upper_body_mask_group);
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
