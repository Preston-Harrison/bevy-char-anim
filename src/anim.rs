use bevy::{
    animation::{AnimationTarget, RepeatAnimation},
    asset::AssetPath,
    prelude::*,
    utils::HashMap,
};

use crate::utils::*;

const LOWER_BODY_MASK_GROUP: u32 = 1;
const LOWER_BODY_MASK: u64 = 1 << LOWER_BODY_MASK_GROUP;
const UPPER_BODY_MASK_GROUP: u32 = 2;
const UPPER_BODY_MASK: u64 = 1 << UPPER_BODY_MASK_GROUP;

const PLAYER_ANIM_INDICES: [&str; 8] = [
    "Idle",
    "IdleUpper",
    "StrafeLeft",
    "StrafeRight",
    "StrafeLeft.001",
    "TPose",
    "WalkBackward",
    "WalkForward",
];

struct PlayerAnimationPaths {
    forward: AssetPath<'static>,
    back: AssetPath<'static>,
    idle: AssetPath<'static>,
    left: AssetPath<'static>,
    right: AssetPath<'static>,
    jump: AssetPath<'static>,
    falling: AssetPath<'static>,
    land: AssetPath<'static>,
}

impl Default for PlayerAnimationPaths {
    fn default() -> Self {
        let get_anim = |name| {
            for (i, curr) in PLAYER_ANIM_INDICES.iter().enumerate() {
                if name == *curr {
                    return GltfAssetLabel::Animation(i).from_asset("models/gltf/character.glb");
                }
            }
            panic!("no anim with name {}", name);
        };

        Self {
            idle: get_anim("Idle"),
            back: get_anim("WalkBackward"),
            forward: get_anim("WalkForward"),
            left: get_anim("StrafeLeft"),
            right: get_anim("StrafeRight"),
            jump: get_anim("TPose"),
            falling: get_anim("TPose"),
            land: get_anim("TPose"),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum AnimationName {
    IdleLowerBody,
    IdleUpperBody,
    Forward,
    Back,
    Left,
    Right,
    Jump,
    Falling,
    Land,
}

impl AnimationName {
    pub fn is_lower_body(&self) -> bool {
        match self {
            Self::IdleUpperBody => false,
            _ => true,
        }
    }

    pub fn get_default_speed(&self) -> f32 {
        1.0
    }

    pub fn get_default_repeat(&self) -> RepeatAnimation {
        match self {
            Self::Jump | Self::Land => RepeatAnimation::Never,
            _ => RepeatAnimation::Forever,
        }
    }
}

pub struct PlayerAnimations {
    anims: HashMap<AnimationName, AnimationNodeIndex>,
}

impl PlayerAnimations {
    pub fn get(&self, id: AnimationName) -> AnimationNodeIndex {
        *self.anims.get(&id).unwrap()
    }

    pub fn get_name(&self, index: AnimationNodeIndex) -> AnimationName {
        self.anims
            .iter()
            .find(|(_, ix)| **ix == index)
            .map(|(name, _)| *name)
            .unwrap()
    }
}

#[derive(Component)]
pub struct PlayerProceduralAnimationTargets {
    pub spine1: Entity,
    pub gun: Entity,
}

pub fn load_player_animations(
    entity: Entity,
    asset_server: &AssetServer,
    children: &Query<&Children>,
    names: &Query<&Name>,
    animation_targets: &Query<&AnimationTarget>,
) -> (
    PlayerAnimations,
    PlayerProceduralAnimationTargets,
    AnimationGraph,
) {
    let player_anims = PlayerAnimationPaths::default();
    let mut graph = AnimationGraph::new();
    let add_node = graph.add_additive_blend(1.0, graph.root);
    let lower_body_blend = graph.add_blend(1.0, add_node);
    let upper_body_blend = graph.add_blend(1.0, add_node);

    let forward_clip = asset_server.load(player_anims.forward.clone());
    let forward_ix = graph.add_clip_with_mask(forward_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);
    let forward = (AnimationName::Forward, forward_ix);

    let back_clip = asset_server.load(player_anims.back.clone());
    let back_ix = graph.add_clip_with_mask(back_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);
    let back = (AnimationName::Back, back_ix);

    let idle_clip = asset_server.load(player_anims.idle.clone());
    let idle_upper_ix =
        graph.add_clip_with_mask(idle_clip.clone(), LOWER_BODY_MASK, 1.0, upper_body_blend);
    let idle_lower_ix = graph.add_clip_with_mask(idle_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);
    let idle_upper = (AnimationName::IdleUpperBody, idle_upper_ix);
    let idle_lower = (AnimationName::IdleLowerBody, idle_lower_ix);

    let left_clip = asset_server.load(player_anims.left.clone());
    let left_ix = graph.add_clip_with_mask(left_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);
    let left = (AnimationName::Left, left_ix);

    let right_clip = asset_server.load(player_anims.right.clone());
    let right_ix = graph.add_clip_with_mask(right_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);
    let right = (AnimationName::Right, right_ix);

    let jump_clip = asset_server.load(player_anims.jump.clone());
    let jump_ix = graph.add_clip_with_mask(jump_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);
    let jump = (AnimationName::Jump, jump_ix);

    let falling_clip = asset_server.load(player_anims.falling.clone());
    let falling_ix = graph.add_clip_with_mask(falling_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);
    let falling = (AnimationName::Falling, falling_ix);

    let land_clip = asset_server.load(player_anims.land.clone());
    let land_ix = graph.add_clip_with_mask(land_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);
    let land = (AnimationName::Land, land_ix);

    let proc_targets =
        init_mixamo_rig_masks(entity, &mut graph, &children, &names, &animation_targets);

    let anims = PlayerAnimations {
        anims: [
            forward, back, idle_upper, idle_lower, left, right, jump, falling, land,
        ]
        .into_iter()
        .collect(),
    };

    (anims, proc_targets, graph)
}

/// groups. The param `entity` should be the entity with an AnimationPlayer.
fn init_mixamo_rig_masks(
    root: Entity,
    graph: &mut AnimationGraph,
    children: &Query<&Children>,
    names: &Query<&Name>,
    animation_targets: &Query<&AnimationTarget>,
) -> PlayerProceduralAnimationTargets {
    // Joints to mask out. All decendants (and this one) will be masked out.
    let upper_body_joint_paths =
        ["mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1/mixamorig:Spine2"];
    let lower_body_joint_paths = [
        "mixamorig:Hips/mixamorig:LeftUpLeg",
        "mixamorig:Hips/mixamorig:RightUpLeg",
    ];
    // Isolates don't mask decendants.
    let isolated_lower_body_joint_paths = ["mixamorig:Hips", "mixamorig:Hips/mixamorig:Spine"];
    let isolated_fully_mask = ["mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1"];

    for joint_path in isolated_fully_mask {
        let entity = find_child_by_path(root, joint_path, &children, &names)
            .expect("upper body joint not found");
        let target = animation_targets.get(entity).expect(&format!(
            "isolate {} should have animation target",
            joint_path
        ));
        graph.add_target_to_mask_group(target.id, LOWER_BODY_MASK_GROUP);
        graph.add_target_to_mask_group(target.id, UPPER_BODY_MASK_GROUP);
    }

    for joint_path in upper_body_joint_paths {
        // Find entity from joint path.
        let upper_body_joint_entity = find_child_by_path(root, joint_path, &children, &names)
            .expect("upper body joint not found");

        // Add every joint for every decendant (including the joint path).
        let entities_to_mask = get_all_descendants(upper_body_joint_entity, &children);
        let targets_to_mask = map_query(entities_to_mask, &animation_targets);
        for target in targets_to_mask {
            graph.add_target_to_mask_group(target.id, UPPER_BODY_MASK_GROUP);
        }
    }

    // Same thing here for both legs.
    for joint_path in lower_body_joint_paths {
        let lower_body_joint_entity = find_child_by_path(root, joint_path, &children, &names)
            .expect("lower body joint not found");

        let entities_to_mask = get_all_descendants(lower_body_joint_entity, &children);
        let targets_to_mask = map_query(entities_to_mask, &animation_targets);
        for target in targets_to_mask.iter() {
            graph.add_target_to_mask_group(target.id, LOWER_BODY_MASK_GROUP);
        }
    }

    // The root of the character (mixamorig:Hips) is still animated by both upper and
    // lower. It is bad to have the same target animated twice by an additive node. Here
    // we decide to assign the hip bone (but not decendants, which we already assigned to
    // either upper or lower) to the lower body.
    for isolate in isolated_lower_body_joint_paths {
        let entity = find_child_by_path(root, isolate, &children, &names)
            .expect(&format!("isolate bone {} should exist", isolate));
        let target = animation_targets
            .get(entity)
            .expect(&format!("isolate {} should have animation target", isolate));
        graph.add_target_to_mask_group(target.id, LOWER_BODY_MASK_GROUP);
    }

    PlayerProceduralAnimationTargets {
        spine1: find_child_by_path(
            root,
            "mixamorig:Hips/mixamorig:Spine/mixamorig:Spine1",
            children,
            names,
        )
        .expect("spine1 not found"),
        gun: find_child_with_name(root, "BlasterN", children, names).expect("shotgun not found"),
    }
}
