use std::time::Duration;

use bevy::animation::RepeatAnimation;
use bevy::asset::AssetPath;
use bevy::prelude::*;

use crate::utils;

pub struct MutantPlugin;

impl Plugin for MutantPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_mutant);
        app.add_systems(Update, (setup_mutant_animations, animate_mutant));
    }
}

struct MutantAnimationPaths {
    attack: AssetPath<'static>,
    idle: AssetPath<'static>,
}

impl Default for MutantAnimationPaths {
    fn default() -> Self {
        const MUTANT_ANIMS: [&str; 3] = ["Attack", "Idle", "TPose"];

        let get_anim = |name| {
            for (i, curr) in MUTANT_ANIMS.iter().enumerate() {
                if name == *curr {
                    return GltfAssetLabel::Animation(i).from_asset("models/gltf/mutant.glb");
                }
            }
            panic!("no anim with name {}", name);
        };

        Self {
            attack: get_anim("Attack"),
            idle: get_anim("Idle"),
        }
    }
}

#[derive(Component)]
struct Mutant;

#[derive(Component)]
struct MutantAnimationPlayer {
    idle: AnimationNodeIndex,
    attack: AnimationNodeIndex,
}

fn spawn_mutant(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        SceneRoot(asset_server.load("models/gltf/mutant.glb#Scene0")),
        Name::new("Mutant"),
        Transform::from_xyz(5.0, 0.0, 5.0),
        Mutant,
    ));
}

fn setup_mutant_animations(
    mut commands: Commands,
    mut query: Query<Entity, Added<AnimationPlayer>>,
    mutants: Query<&Mutant>,
    parents: Query<&ChildOf>,
    asset_server: Res<AssetServer>,
    mut animation_graphs: ResMut<Assets<AnimationGraph>>,
) {
    for entity in query.iter_mut() {
        if utils::find_upwards(entity, &parents, &mutants).is_none() {
            continue;
        }

        let mut graph = AnimationGraph::new();
        let paths = MutantAnimationPaths::default();

        let idle_clip = asset_server.load(paths.idle);
        let idle_ix = graph.add_clip(idle_clip, 1.0, graph.root);

        let attack_clip = asset_server.load(paths.attack);
        let attack_ix = graph.add_clip(attack_clip, 1.0, graph.root);

        commands
            .entity(entity)
            .insert(MutantAnimationPlayer {
                idle: idle_ix,
                attack: attack_ix,
            })
            .insert(AnimationTransitions::new())
            .insert(AnimationGraphHandle(animation_graphs.add(graph)));
    }
}

fn animate_mutant(
    mut query: Query<(
        &mut AnimationPlayer,
        &mut AnimationTransitions,
        &MutantAnimationPlayer,
    )>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    for (mut player, mut transitions, anims) in query.iter_mut() {
        if keys.just_pressed(KeyCode::Digit1) {
            transitions
                .play(&mut player, anims.attack, Duration::from_millis(1000))
                .set_repeat(RepeatAnimation::Never);
        }
        if player.all_finished() {
            transitions.play(&mut player, anims.idle, Duration::from_millis(1000));
        }
    }
}
