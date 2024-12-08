use bevy::{
    animation::{AnimationTarget, RepeatAnimation}, asset::AssetPath, prelude::*
};

use crate::utils::*;

const LOWER_BODY_MASK_GROUP: u32 = 1;
const LOWER_BODY_MASK: u64 = 1 << LOWER_BODY_MASK_GROUP;
const UPPER_BODY_MASK_GROUP: u32 = 2;
const UPPER_BODY_MASK: u64 = 1 << UPPER_BODY_MASK_GROUP;

const PLAYER_ANIM_INDICES: [&str; 12] = [
    "AimingIdle",
    "FallingIdle",
    "FallToLand",
    "FiringRifle",
    "Jump",
    "RifleRun",
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
pub struct PlayerAnimationIndicies {
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

#[derive(Component)]
pub struct PlayerProceduralAnimationTargets {
    spine: Entity,
}

pub fn load_player_animations(
    entity: Entity,
    asset_server: &AssetServer,
    children: &Query<&Children>,
    names: &Query<&Name>,
    animation_targets: &Query<&AnimationTarget>,
) -> (PlayerAnimationIndicies, PlayerProceduralAnimationTargets, AnimationGraph) {
    let player_anims = PlayerAnimations::default();
    let mut graph = AnimationGraph::new();
    let add_node = graph.add_additive_blend(1.0, graph.root);
    let lower_body_blend = graph.add_blend(1.0, add_node);
    let upper_body_blend = graph.add_blend(1.0, add_node);

    let forward_clip = asset_server.load(player_anims.forward.clone());
    let forward = graph.add_clip_with_mask(forward_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);

    let back_clip = asset_server.load(player_anims.back.clone());
    let back = graph.add_clip_with_mask(back_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);

    let rifle_clip_upper = asset_server.load(player_anims.idle_upper.clone());
    let idle_upper =
        graph.add_clip_with_mask(rifle_clip_upper, LOWER_BODY_MASK, 1.0, upper_body_blend);

    let rifle_clip_lower = asset_server.load(player_anims.idle_lower.clone());
    let idle_lower =
        graph.add_clip_with_mask(rifle_clip_lower, UPPER_BODY_MASK, 1.0, lower_body_blend);

    let left_clip = asset_server.load(player_anims.left.clone());
    let left = graph.add_clip_with_mask(left_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);

    let right_clip = asset_server.load(player_anims.right.clone());
    let right = graph.add_clip_with_mask(right_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);

    let jump_clip = asset_server.load(player_anims.jump.clone());
    let jump = graph.add_clip_with_mask(jump_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);

    let falling_clip = asset_server.load(player_anims.falling.clone());
    let falling = graph.add_clip_with_mask(falling_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);

    let land_clip = asset_server.load(player_anims.land.clone());
    let land = graph.add_clip_with_mask(land_clip, UPPER_BODY_MASK, 1.0, lower_body_blend);

    let proc_targets = init_mixamo_rig_masks(
        entity,
        &mut graph,
        &children,
        &names,
        &animation_targets,
    );

    let indices = PlayerAnimationIndicies {
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
    };

    (indices, proc_targets, graph)
}

/// groups. The param `entity` should be the entity with an AnimationPlayer.
fn init_mixamo_rig_masks(
    entity: Entity,
    graph: &mut AnimationGraph,
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
        graph.add_target_to_mask_group(target.id, LOWER_BODY_MASK_GROUP);
        graph.add_target_to_mask_group(target.id, UPPER_BODY_MASK_GROUP);

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
            graph.add_target_to_mask_group(target.id, UPPER_BODY_MASK_GROUP);
        }
    }

    // Same thing here for both legs.
    for joint_path in lower_body_joint_paths {
        let lower_body_joint_entity = find_child_by_path(entity, joint_path, &children, &names)
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
        let entity = find_child_by_path(entity, isolate, &children, &names)
            .expect(&format!("isolate bone {} should exist", isolate));
        let target = animation_targets
            .get(entity)
            .expect(&format!("isolate {} should have animation target", isolate));
        graph.add_target_to_mask_group(target.id, LOWER_BODY_MASK_GROUP);
    }

    proc_targets.expect("proc target to be set")
}
