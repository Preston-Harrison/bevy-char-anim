use bevy::{
    animation::{AnimationTarget, RepeatAnimation},
    asset::AssetPath,
    prelude::*,
    utils::HashMap,
};

use crate::utils::*;

pub struct AnimationPlugin;

impl Plugin for AnimationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, cancel_translation);
    }
}

const LOWER_BODY_MASK_GROUP: u32 = 1;
const LOWER_BODY_MASK: u64 = 1 << LOWER_BODY_MASK_GROUP;
const UPPER_BODY_MASK_GROUP: u32 = 2;
const UPPER_BODY_MASK: u64 = 1 << UPPER_BODY_MASK_GROUP;

const PLAYER_ANIM_INDICES: [&str; 10] = [
    "FallingIdle",
    "Idle",
    "IdleUpper",
    "JumpDown",
    "StrafeLeft",
    "StrafeRight",
    "TPose",
    "TurnLeft45",
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
            jump: get_anim("FallingIdle"),
            falling: get_anim("FallingIdle"),
            land: get_anim("JumpDown"),
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
    pub bullet_point: Entity,
}

pub fn load_player_animations(
    entity: Entity,
    asset_server: &AssetServer,
    children: &Query<&Children>,
    parents: &Query<&Parent>,
    transforms: &Query<&Transform>,
    commands: Commands,
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

    let proc_targets = init_mixamo_rig_masks(
        entity,
        &mut graph,
        &children,
        &parents,
        &names,
        &transforms,
        &animation_targets,
        commands,
    );

    let anims = PlayerAnimations {
        anims: [
            forward, back, idle_upper, idle_lower, left, right, jump, falling, land,
        ]
        .into_iter()
        .collect(),
    };

    (anims, proc_targets, graph)
}

#[derive(PartialEq)]
enum Mask {
    Upper,
    Lower,
    Both,
}

/// groups. The param `entity` should be the entity with an AnimationPlayer.
fn init_mixamo_rig_masks(
    root: Entity,
    graph: &mut AnimationGraph,
    children: &Query<&Children>,
    parents: &Query<&Parent>,
    names: &Query<&Name>,
    transforms: &Query<&Transform>,
    animation_targets: &Query<&AnimationTarget>,
    mut commands: Commands,
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
            if *mask_type == Mask::Lower || *mask_type == Mask::Both {
                graph.add_target_to_mask_group(target.id, LOWER_BODY_MASK_GROUP);
            }
            if *mask_type == Mask::Upper || *mask_type == Mask::Both {
                graph.add_target_to_mask_group(target.id, UPPER_BODY_MASK_GROUP);
            }
        }
    }

    let hips = find_child_with_name(root, "mixamorig:Hips", children, names).unwrap();
    let hip_parent = parents.get(hips).unwrap();
    let canceller = commands
        .spawn(TranslationCanceller {
            offset: transforms.get(hips).unwrap().translation,
            enabled: false,
        })
        .id();
    commands.entity(hips).set_parent(canceller);
    commands.entity(canceller).set_parent(hip_parent.get());

    PlayerProceduralAnimationTargets {
        spine1: find_child_with_name(root, "mixamorig:Spine1", children, names).unwrap(),
        bullet_point: find_child_with_name(root, "BlasterN", children, names).unwrap(),
    }
}

#[derive(Component)]
#[require(Transform)]
struct TranslationCanceller {
    offset: Vec3,
    enabled: bool,
}

fn cancel_translation(
    mut cancellers: Query<
        (&mut Transform, &Children, &TranslationCanceller),
        With<TranslationCanceller>,
    >,
    transforms: Query<&Transform, Without<TranslationCanceller>>,
) {
    for (mut transform, children, canceller) in cancellers.iter_mut() {
        if !canceller.enabled {
            continue;
        };
        if children.len() > 1 {
            warn!("cannot cancel translation for more than one child");
            continue;
        };
        let Some(child) = children.get(0) else {
            continue;
        };
        let Ok(child_transform) = transforms.get(*child) else {
            warn!("child of translation canceller has no transform");
            continue;
        };
        transform.translation = -child_transform.translation + canceller.offset;
    }
}
