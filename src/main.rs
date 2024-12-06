use bevy::{animation::AnimationTarget, prelude::*};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, update)
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
    ));

    // Spawn the ground.
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(7.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
}

fn update(
    mut commands: Commands,
    mut new_anim_players: Query<(Entity, &mut AnimationPlayer), Added<AnimationPlayer>>,
    asset_server: Res<AssetServer>,
    children: Query<&Children>,
    names: Query<&Name>,
    mut animation_graphs: ResMut<Assets<AnimationGraph>>,
    animation_targets: Query<&AnimationTarget>,
) {
    for (entity, mut player) in new_anim_players.iter_mut() {
        // Actual mask is a bitmap, but mask group is nth bit in bitmap.
        let upper_body_mask_group = 1;
        let upper_body_mask = 1 << upper_body_mask_group;
        // Joint to mask out. All decendants (and this one) will be masked out.
        let upper_body_joint_path = "mixamorig:Hips/mixamorig:Spine";

        // Same thing for lower body
        let lower_body_mask_group = 2;
        let lower_body_mask = 1 << lower_body_mask_group;
        let lower_body_joint_paths = [
            "mixamorig:Hips/mixamorig:LeftUpLeg",
            "mixamorig:Hips/mixamorig:RightUpLeg",
        ];

        let hip_path = "mixamorig:Hips";

        let mut graph = AnimationGraph::new();
        let add_node = graph.add_additive_blend(1.0, graph.root);

        // Load walk forward and rifle idle animations.
        let forward_anim_path = GltfAssetLabel::Animation(2).from_asset("character.glb");
        let forward_clip = asset_server.load(forward_anim_path);
        let forward = graph.add_clip_with_mask(forward_clip, upper_body_mask, 1.0, add_node);
        let rifle_anim_path = GltfAssetLabel::Animation(0).from_asset("character.glb");
        let rifle_clip = asset_server.load(rifle_anim_path);
        let rifle_idle = graph.add_clip_with_mask(rifle_clip, lower_body_mask, 1.0, add_node);

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
        let hip =
            find_child_by_path(entity, hip_path, &children, &names).expect("hip bone should exist");
        let hip_target = animation_targets
            .get(hip)
            .expect("hip should have animation target");
        graph.add_target_to_mask_group(hip_target.id, lower_body_mask_group);

        commands
            .entity(entity)
            .insert(AnimationGraphHandle(animation_graphs.add(graph)));

        player.play(forward).repeat();
        player.play(rifle_idle).repeat();
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
fn map_query<T: Component + Clone + 'static>(entites: Vec<Entity>, query: &Query<&T>) -> Vec<T> {
    entites
        .into_iter()
        .flat_map(|v| query.get(v).ok())
        .cloned()
        .collect::<Vec<_>>()
}
