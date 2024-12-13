use bevy::{
    animation::{ActiveAnimation, AnimationTarget, RepeatAnimation},
    asset::AssetPath,
    prelude::*,
    utils::HashMap,
};

use crate::{
    state::{run_player_animations, AnimationNodes},
    utils::*,
};

pub struct AnimationPlugin;

impl Plugin for AnimationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, run_player_animations);
    }
}

/// Masks out bones in the lower body.
const LOWER_BODY_MASK_GROUP: u32 = 1;
const LOWER_BODY_MASK: u64 = 1 << LOWER_BODY_MASK_GROUP;
/// Masks out bones in the upper body.
const UPPER_BODY_MASK_GROUP: u32 = 2;
const UPPER_BODY_MASK: u64 = 1 << UPPER_BODY_MASK_GROUP;

struct PlayerAnimationPaths {
    forward: AssetPath<'static>,
    back: AssetPath<'static>,
    idle: AssetPath<'static>,
    left: AssetPath<'static>,
    right: AssetPath<'static>,
    jump: AssetPath<'static>,
    falling: AssetPath<'static>,
    land: AssetPath<'static>,
    sprint: AssetPath<'static>,
}

impl Default for PlayerAnimationPaths {
    fn default() -> Self {
        /// Animation indices in order according to character gltf file.
        const PLAYER_ANIM_INDICES: [&str; 11] = [
            "FallingIdle",
            "Idle",
            "IdleUpper",
            "JumpDown",
            "RifleRun",
            "StrafeLeft",
            "StrafeRight",
            "TPose",
            "TurnLeft45",
            "WalkBackward",
            "WalkForward",
        ];

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
            jump: get_anim("FallingIdle"),
            falling: get_anim("FallingIdle"),
            land: get_anim("JumpDown"),
            sprint: get_anim("RifleRun"),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
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
    Sprint,
}

impl AnimationName {
    /// Returns whether this animation affects the lower body of the player.
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

    pub fn apply_defaults<'a>(
        &self,
        index: AnimationNodeIndex,
        anim: &'a mut ActiveAnimation,
    ) -> &'a mut ActiveAnimation {
        let name = self.get_name(index);
        anim.set_speed(name.get_default_speed())
            .set_repeat(name.get_default_repeat())
    }
}

pub struct PlayerProceduralAnimationTargets {
    pub spine1: Entity,
    pub bullet_point: Entity,
}

pub fn load_player_animations(
    entity: Entity,
    asset_server: &AssetServer,
    children: &Query<&Children>,
    names: &Query<&Name>,
    animation_targets: &Query<&AnimationTarget>,
    commands: Commands,
    parents: &Query<&Parent>,
) -> (
    PlayerAnimations,
    PlayerProceduralAnimationTargets,
    AnimationGraph,
    AnimationNodes,
) {
    let player_anims = PlayerAnimationPaths::default();
    let mut graph = AnimationGraph::new();
    let add_node = graph.add_additive_blend(1.0, graph.root);
    let lower_body_blend = graph.add_blend(1.0, add_node);
    let upper_body_blend = graph.add_blend(1.0, add_node);
    let full_body = graph.add_blend(1.0, graph.root);

    let nodes = AnimationNodes {
        upper_lower_add: add_node,
        full_body,
    };

    let mut anims = HashMap::default();

    let mut add_anim = |name, path: &AssetPath<'static>, mask, parent| {
        let forward_clip = asset_server.load(path.clone());
        let forward_ix = graph.add_clip_with_mask(forward_clip, mask, 1.0, parent);
        anims.insert(name, forward_ix);
    };

    add_anim(
        AnimationName::Forward,
        &player_anims.forward,
        UPPER_BODY_MASK,
        lower_body_blend,
    );
    add_anim(
        AnimationName::Back,
        &player_anims.back,
        UPPER_BODY_MASK,
        lower_body_blend,
    );
    add_anim(
        AnimationName::IdleUpperBody,
        &player_anims.idle,
        LOWER_BODY_MASK,
        upper_body_blend,
    );
    add_anim(
        AnimationName::IdleLowerBody,
        &player_anims.idle,
        UPPER_BODY_MASK,
        lower_body_blend,
    );
    add_anim(
        AnimationName::Left,
        &player_anims.left,
        UPPER_BODY_MASK,
        lower_body_blend,
    );
    add_anim(
        AnimationName::Right,
        &player_anims.right,
        UPPER_BODY_MASK,
        lower_body_blend,
    );
    add_anim(
        AnimationName::Jump,
        &player_anims.jump,
        UPPER_BODY_MASK,
        lower_body_blend,
    );
    add_anim(
        AnimationName::Falling,
        &player_anims.falling,
        UPPER_BODY_MASK,
        lower_body_blend,
    );
    add_anim(
        AnimationName::Land,
        &player_anims.land,
        UPPER_BODY_MASK,
        lower_body_blend,
    );
    add_anim(
        AnimationName::Sprint,
        &player_anims.sprint,
        // No mask, notably this affects procedural bones as well.
        0,
        full_body,
    );

    let anims = PlayerAnimations { anims };

    let proc_targets = init_mixamo_rig_masks(
        entity,
        &mut graph,
        children,
        names,
        animation_targets,
        commands,
        parents,
    );

    (anims, proc_targets, graph, nodes)
}

#[derive(PartialEq)]
enum Mask {
    Upper,
    Lower,
    Both,
}

fn init_mixamo_rig_masks(
    root: Entity,
    graph: &mut AnimationGraph,
    children: &Query<&Children>,
    names: &Query<&Name>,
    animation_targets: &Query<&AnimationTarget>,
    mut commands: Commands,
    parents: &Query<&Parent>,
) -> PlayerProceduralAnimationTargets {
    // (name, should masks decendants, mask type)
    let masks = &[
        ("mixamorig:Spine2", true, Mask::Upper),
        ("mixamorig:LeftUpLeg", true, Mask::Lower),
        ("mixamorig:RightUpLeg", true, Mask::Lower),
        ("mixamorig:Spine", false, Mask::Lower),
        ("mixamorig:Hips", false, Mask::Lower),
        ("mixamorig:Spine1", false, Mask::Both),
    ];

    for (name, mask_decendants, mask_type) in masks {
        let entity = find_child_with_name(root, name, &children, &names)
            .expect("upper body joint not found");
        let target = animation_targets
            .get(entity)
            .expect(&format!("isolate {} should have animation target", name));

        let targets = if *mask_decendants {
            let entities_to_mask = get_all_descendants(entity, &children);
            map_query(entities_to_mask, &animation_targets)
        } else {
            vec![*target]
        };
        for target in targets {
            if *mask_type == Mask::Lower {
                graph.add_target_to_mask_group(target.id, LOWER_BODY_MASK_GROUP);
            }
            if *mask_type == Mask::Upper {
                graph.add_target_to_mask_group(target.id, UPPER_BODY_MASK_GROUP);
            }
            if *mask_type == Mask::Both {
                graph.add_target_to_mask_group(target.id, LOWER_BODY_MASK_GROUP);
                graph.add_target_to_mask_group(target.id, UPPER_BODY_MASK_GROUP);
            }
        }
    }

    let spine1_proc_target = commands
        .spawn((Transform::default(), Visibility::default()))
        .id();

    let spine1 = find_child_with_name(root, "mixamorig:Spine1", children, names).unwrap();
    let spine1_parent = parents.get(spine1).unwrap().get();
    commands
        .entity(spine1_proc_target)
        .set_parent(spine1_parent);
    commands.entity(spine1).set_parent(spine1_proc_target);

    PlayerProceduralAnimationTargets {
        spine1: spine1_proc_target,
        bullet_point: find_child_with_name(root, "BlasterN", children, names).unwrap(),
    }
}
