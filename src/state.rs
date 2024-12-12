use std::f32::consts::FRAC_PI_2;

use bevy::animation::ActiveAnimation;
use bevy::prelude::*;

use crate::anim::{AnimationName, PlayerAnimations, PlayerProceduralAnimationTargets};
use crate::utils::{self, *};
use crate::Player;

pub fn run_player_animations(
    mut states: Query<(Entity, &mut PlayerAnimationState, &mut AnimationPlayer, &AnimationGraphHandle)>,
    parents: Query<&Parent>,
    players: Query<&Player>,
    mut transforms: Query<&mut Transform>,
    global_transforms: Query<&GlobalTransform>,
    mut anim_graphs: ResMut<Assets<AnimationGraph>>,
) {
    for (entity, mut state, mut player, graph) in states.iter_mut() {
        let Some(graph) = anim_graphs.get_mut(graph) else {
            continue;
        };
        state.transition(&player);
        state.update_player(&mut player, graph);
        let Some((root_entity, _)) = utils::find_upwards(entity, &parents, &players) else {
            error!("player animation state not attached to child of player");
            continue;
        };
        state.update_transforms(root_entity, &mut transforms, &global_transforms);
        state.input = None;
    }
}

#[derive(Default)]
pub struct PlayerAnimationInput {
    /// +Y is forward
    pub local_movement_direction: Vec2,
    pub is_sprinting: bool,
    pub look_y: f32,
    pub look_x: f32,

    pub just_jumped: bool,
    pub is_grounded: bool,
}

#[derive(Component)]
pub struct PlayerAnimationState {
    anims: PlayerAnimations,
    lower_body: LowerBodyState,
    input: Option<PlayerAnimationInput>,
    pub proc_targets: PlayerProceduralAnimationTargets,

    lower_body_y: f32,
    lower_body_target_y: f32,
    upper_body_y: f32,

    is_sprinting: bool,
    nodes: AnimationNodes,
}

pub struct AnimationNodes {
    /// The node that adds upper and lower body anims.
    pub upper_lower_add: AnimationNodeIndex,
    /// The node that blends between `upper_lower_add` and full body animations.
    pub full_body: AnimationNodeIndex,
}

impl PlayerAnimationState {
    pub fn new(
        anims: PlayerAnimations,
        proc_targets: PlayerProceduralAnimationTargets,
        nodes: AnimationNodes,
    ) -> Self {
        Self {
            anims,
            lower_body: LowerBodyState::Idle,
            input: None,
            proc_targets,

            lower_body_y: 0.0,
            lower_body_target_y: 0.0,
            upper_body_y: 0.0,
            is_sprinting: false,
            nodes,
        }
    }
}

impl PlayerAnimationState {
    pub fn set_input(&mut self, input: PlayerAnimationInput) {
        self.input = Some(input);
    }

    pub fn transition(&mut self, player: &AnimationPlayer) {
        let Some(ref input) = self.input else {
            return;
        };
        let is_finished = |anim| player.animation(anim).unwrap().is_finished();

        self.is_sprinting = input.is_sprinting;
        self.lower_body = match self.lower_body {
            LowerBodyState::Land => {
                if is_finished(self.anims.get(AnimationName::Land)) {
                    LowerBodyState::Idle
                } else {
                    LowerBodyState::Land
                }
            }
            LowerBodyState::Jump => {
                // No jump anim for now as it looks slow.
                LowerBodyState::Falling
            }
            LowerBodyState::Falling => {
                if input.is_grounded {
                    LowerBodyState::Land
                } else {
                    LowerBodyState::Falling
                }
            }
            _ if input.just_jumped => LowerBodyState::Jump,
            _ if input.local_movement_direction.length() < 0.1 => LowerBodyState::Idle,
            _ => *sample_cardinal(
                &[
                    LowerBodyState::Forward,
                    LowerBodyState::Back,
                    LowerBodyState::Left,
                    LowerBodyState::Right,
                ],
                input.local_movement_direction,
            ),
        };
    }

    pub fn update_player(&self, player: &mut AnimationPlayer, graph: &mut AnimationGraph) {
        let rate = 0.9;
        let threshold = 0.01;

        // If is sprinting, fade out other anims and just play full body sprint
        // animation.
        if self.is_sprinting {
            let sprint_anim = self.anims.get(AnimationName::Sprint);
            player
                .play(sprint_anim)
                .set_speed(AnimationName::Sprint.get_default_speed())
                .set_repeat(AnimationName::Sprint.get_default_repeat());
            
            let upper_lower_add = graph.get_mut(self.nodes.upper_lower_add).unwrap();
            upper_lower_add.weight = upper_lower_add.weight * rate;
            if upper_lower_add.weight < threshold {
                upper_lower_add.weight = 0.0;
            }

            let full_body = graph.get_mut(self.nodes.full_body).unwrap();
            full_body.weight = (full_body.weight / rate).clamp(threshold, 1.0);

            return;
        }

        let full_body = graph.get_mut(self.nodes.full_body).unwrap();
        full_body.weight = full_body.weight * rate;
        if full_body.weight < threshold {
            full_body.weight = 0.0;
        }

        let upper_lower_add = graph.get_mut(self.nodes.upper_lower_add).unwrap();
        upper_lower_add.weight = (upper_lower_add.weight / rate).clamp(threshold, 1.0);

        let target_lower_body_anim = match self.lower_body {
            LowerBodyState::Idle => self.anims.get(AnimationName::IdleLowerBody),
            LowerBodyState::Forward => self.anims.get(AnimationName::Forward),
            LowerBodyState::Back => self.anims.get(AnimationName::Back),
            LowerBodyState::Right => self.anims.get(AnimationName::Right),
            LowerBodyState::Left => self.anims.get(AnimationName::Left),
            LowerBodyState::Jump => self.anims.get(AnimationName::Jump),
            LowerBodyState::Falling => self.anims.get(AnimationName::Falling),
            LowerBodyState::Land => self.anims.get(AnimationName::Land),
        };

        let animations_to_fade = player
            .playing_animations()
            .filter_map(|(ix, _)| {
                if self.anims.get_name(*ix).is_lower_body() && *ix != target_lower_body_anim {
                    Some(*ix)
                } else {
                    None
                }
            })
            .collect();

        fade_out_animations(player, animations_to_fade, rate, threshold);
        fade_in_animation(player, target_lower_body_anim, 1.0 / rate, threshold)
            .set_speed(
                self.anims
                    .get_name(target_lower_body_anim)
                    .get_default_speed(),
            )
            .set_repeat(
                self.anims
                    .get_name(target_lower_body_anim)
                    .get_default_repeat(),
            );

        let target_upper_body_anim = self.anims.get(AnimationName::IdleUpperBody);
        player
            .play(target_upper_body_anim)
            .set_weight(1.0)
            .set_speed(
                self.anims
                    .get_name(target_upper_body_anim)
                    .get_default_speed(),
            )
            .set_repeat(
                self.anims
                    .get_name(target_upper_body_anim)
                    .get_default_repeat(),
            );
    }

    pub fn update_transforms(
        &mut self,
        root_entity: Entity,
        transforms: &mut Query<&mut Transform>,
        global_transforms: &Query<&GlobalTransform>,
    ) {
        let mut root_local = transforms.get_mut(root_entity).unwrap();
        let Some(ref input) = self.input else {
            warn!("missed input");
            return;
        };

        if input.local_movement_direction.length() < 0.1 && input.is_grounded && !input.is_sprinting
        {
            if self.upper_body_y > 45f32.to_radians() {
                self.lower_body_target_y += 45f32.to_radians();
            } else if self.upper_body_y < -45f32.to_radians() {
                self.lower_body_target_y -= 45f32.to_radians();
            }

            self.lower_body_y = self.lower_body_y.lerp(self.lower_body_target_y, 0.05);
            self.upper_body_y = input.look_y - self.lower_body_y;
        } else {
            self.lower_body_y = self.lower_body_y.lerp(input.look_y, 0.05);
            self.lower_body_target_y = input.look_y;
            self.upper_body_y = input.look_y - self.lower_body_y;
        }

        root_local.rotation = Quat::from_axis_angle(Vec3::Y, self.lower_body_y);

        let mut spine1_local = transforms.get_mut(self.proc_targets.spine1).unwrap();
        let bullet_point_global = global_transforms
            .get(self.proc_targets.bullet_point)
            .unwrap();
        let spine1_global = global_transforms.get(self.proc_targets.spine1).unwrap();
        let root_global = global_transforms.get(root_entity).unwrap();

        rotate_spine_to_x(
            root_global,
            bullet_point_global,
            spine1_global,
            &mut spine1_local,
            input.look_x,
            self.upper_body_y,
        );
    }
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

/// Rotates the spine bone to the target rotation about the player x-axis.
fn rotate_spine_to_x(
    player_global: &GlobalTransform,
    bullet_point_global: &GlobalTransform,
    spine1_global: &GlobalTransform,
    spine1_local: &mut Transform,
    target_gun_x_rotation: f32,
    target_gun_y_rotation: f32,
) {
    if target_gun_x_rotation > FRAC_PI_2 * 0.9 || target_gun_x_rotation < -FRAC_PI_2 * 0.9 {
        warn!("gun x rotation too large");
        return;
    }

    // Compute the target gun rotation in global space.
    let target_vec = Quat::from_axis_angle(Vec3::Y, target_gun_y_rotation)
        * Quat::from_axis_angle(Vec3::X, target_gun_x_rotation)
        * Vec3::Z;
    let global_target = player_global.rotation() * target_vec;

    // Compute the current forward direction of the bullet_point in global space.
    let current_bullet_forward = bullet_point_global.rotation() * Vec3::Z;

    // Compute the rotation needed to align the current bullet_point forward to the target direction.
    let alignment_rotation = Quat::from_rotation_arc(
        spine1_global.rotation().inverse() * current_bullet_forward,
        spine1_global.rotation().inverse() * global_target,
    );

    spine1_local.rotation = spine1_local.rotation * alignment_rotation;
    // Slide the spine vertical if possible.
    spine1_local.rotation = spine1_local
        .rotation
        .rotate_towards(Quat::from_axis_angle(Vec3::Y, 0.0), 0.1);
}
