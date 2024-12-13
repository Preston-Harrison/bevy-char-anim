use std::f32::consts::FRAC_PI_2;

use bevy::animation::ActiveAnimation;
use bevy::prelude::*;

use crate::anim::{AnimationName, PlayerAnimations, PlayerProceduralAnimationTargets};
use crate::utils;
use crate::Player;

pub fn run_player_animations(
    mut states: Query<(
        Entity,
        &mut PlayerAnimationState,
        &mut AnimationPlayer,
        &AnimationGraphHandle,
    )>,
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
        state.update_transforms(root_entity, &mut transforms, &global_transforms, &player);
        state.input = None;
    }
}

/// An authoritative input that changes the animation. This should be valid, e.g.
/// sending is_sprinting with !is_grounded could have weird animation effects if
/// you can't sprint while airborne.
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
    config: AnimationStateConfig,
}

pub struct AnimationStateConfig {
    /// The rate that's used to blend between animations. Must be in (0, 1].
    pub blend_rate: f32,
    /// The threshold under which animations are stopped, and which animations
    /// are started at.
    pub blend_threshold: f32,
    /// Gets the max angle to rotate to get the spine to aim in such a way to
    /// point the bullet point forward. This is used when blending from sprinting
    /// (where the spine is controlled fully by the animation) and a normal pose
    /// (where the spine is controlled fully by the input). 
    pub sprint_reaim_max_angle: fn(Option<&ActiveAnimation>) -> f32,
    /// The elapsed time for which the landing animation is in the impact phase
    /// and can be cancelled to play other animations.
    pub mostly_landed_elapsed_time: f32,
    /// The angle in radians for that makes the player's feet move when turning
    /// while stationary.
    pub stationary_turn_threshold: f32,
    /// The speed at which to lerp the players feet to the new position when turning
    /// while stationary.
    pub stationary_turn_lerp_speed: f32,
    /// The max angle per frame to rotate from the player's look position (which 
    /// is done by rotating the spine) to the sprint animation spine position.
    pub spine1_into_sprint_max_angle: f32,
}

fn sprint_reaim_max_angle(anim: Option<&ActiveAnimation>) -> f32 {
    match anim {
        Some(a) if a.weight() < 0.25 => 0.05,
        Some(_) => 0.0,
        None => 0.1,
    }
}

impl Default for AnimationStateConfig {
    fn default() -> Self {
        Self {
            blend_rate: 0.9,
            blend_threshold: 0.01,
            sprint_reaim_max_angle,
            mostly_landed_elapsed_time: 0.25,
            stationary_turn_threshold: 45f32.to_radians(),
            stationary_turn_lerp_speed: 0.05,
            spine1_into_sprint_max_angle: 0.01,
        }
    }
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
            config: AnimationStateConfig::default(),
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
        let is_finished = |anim| {
            player
                .animation(anim)
                .map(|a| a.is_finished())
                .unwrap_or(true)
        };

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
            _ => *utils::sample_cardinal(
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

    fn get_lower_body_anim_from_state(&self) -> AnimationNodeIndex {
        match self.lower_body {
            LowerBodyState::Idle => self.anims.get(AnimationName::IdleLowerBody),
            LowerBodyState::Forward => self.anims.get(AnimationName::Forward),
            LowerBodyState::Back => self.anims.get(AnimationName::Back),
            LowerBodyState::Right => self.anims.get(AnimationName::Right),
            LowerBodyState::Left => self.anims.get(AnimationName::Left),
            LowerBodyState::Jump => self.anims.get(AnimationName::Jump),
            LowerBodyState::Falling => self.anims.get(AnimationName::Falling),
            LowerBodyState::Land => self.anims.get(AnimationName::Land),
        }
    }

    fn should_play_sprint_anim(&self, player: &AnimationPlayer) -> bool {
        // Sprinting should occur if the player has played the impact part of the
        // landing animation, but sprinting doesn't need to wait for the full animation.
        let mostly_landed = player
            .animation(self.anims.get(AnimationName::Land))
            .is_none_or(|a| {
                a.elapsed() > self.config.mostly_landed_elapsed_time || a.is_finished()
            });
        // Don't play sprint animation if we are in the landing state but the landing
        // animation hasn't started yet.
        let not_played_land_anim_yet = self.lower_body == LowerBodyState::Land
            && player
                .animation(self.anims.get(AnimationName::Land))
                .is_none();
        let not_just_landed = self.lower_body != LowerBodyState::Land || mostly_landed;

        // This statement should read: play the sprinting animation if we are sprinting,
        // haven't just landed, and aren't about to start the landing animation.
        self.is_sprinting && not_just_landed && !not_played_land_anim_yet
    }

    fn fade_in_full_body(&self, graph: &mut AnimationGraph) {
        let rate = self.config.blend_rate;
        let threshold = self.config.blend_threshold;

        let upper_lower_add = graph.get_mut(self.nodes.upper_lower_add).unwrap();
        upper_lower_add.weight = upper_lower_add.weight * rate;
        if upper_lower_add.weight < threshold {
            upper_lower_add.weight = 0.0;
        }

        let full_body = graph.get_mut(self.nodes.full_body).unwrap();
        full_body.weight = (full_body.weight / rate).clamp(threshold, 1.0);
    }

    fn fade_in_split_body(&self, graph: &mut AnimationGraph) {
        let rate = self.config.blend_rate;
        let threshold = self.config.blend_threshold;

        let full_body = graph.get_mut(self.nodes.full_body).unwrap();
        full_body.weight = full_body.weight * rate;
        if full_body.weight < threshold {
            full_body.weight = 0.0;
        }

        let upper_lower_add = graph.get_mut(self.nodes.upper_lower_add).unwrap();
        upper_lower_add.weight = (upper_lower_add.weight / rate).clamp(threshold, 1.0);
    }

    pub fn update_player(&self, player: &mut AnimationPlayer, graph: &mut AnimationGraph) {
        let rate = self.config.blend_rate;
        let threshold = self.config.blend_threshold;

        // If is sprinting, fade out other anims and just play full body sprint
        // animation.
        if self.should_play_sprint_anim(player) {
            let sprint_anim = self.anims.get(AnimationName::Sprint);
            let active_anim = player.play(sprint_anim);
            self.anims.apply_defaults(sprint_anim, active_anim);
            self.fade_in_full_body(graph);
            return;
        }

        self.fade_in_split_body(graph);
        let target_lower_body_anim = self.get_lower_body_anim_from_state();

        // Only fade lower body animations excluding the one that's being faded in.
        let filter = |(ix, _): (&AnimationNodeIndex, &ActiveAnimation)| {
            (self.anims.get_name(*ix).is_lower_body() && *ix != target_lower_body_anim)
                .then_some(*ix)
        };
        let animations_to_fade = player.playing_animations().filter_map(filter).collect();

        fade_out_animations(player, animations_to_fade, rate, threshold);
        let lower_body_anim =
            fade_in_animation(player, target_lower_body_anim, 1.0 / rate, threshold);
        self.anims
            .apply_defaults(target_lower_body_anim, lower_body_anim);

        let target_upper_body_anim = self.anims.get(AnimationName::IdleUpperBody);
        let active_anim = player.play(target_upper_body_anim).set_weight(1.0);
        self.anims
            .apply_defaults(target_upper_body_anim, active_anim);
    }

    pub fn update_transforms(
        &mut self,
        root_entity: Entity,
        transforms: &mut Query<&mut Transform>,
        global_transforms: &Query<&GlobalTransform>,
        player: &AnimationPlayer,
    ) {
        let mut root_local = transforms.get_mut(root_entity).unwrap();
        let Some(ref input) = self.input else {
            warn!("missed input");
            return;
        };

        if input.local_movement_direction.length() < 0.1 && input.is_grounded && !input.is_sprinting
        {
            if self.upper_body_y > self.config.stationary_turn_threshold {
                self.lower_body_target_y += self.config.stationary_turn_threshold;
            } else if self.upper_body_y < -self.config.stationary_turn_threshold {
                self.lower_body_target_y -= self.config.stationary_turn_threshold;
            }

            self.lower_body_y = self.lower_body_y.lerp(
                self.lower_body_target_y,
                self.config.stationary_turn_lerp_speed,
            );
            self.upper_body_y = input.look_y - self.lower_body_y;
        } else {
            self.lower_body_y = self
                .lower_body_y
                .lerp(input.look_y, self.config.stationary_turn_lerp_speed);
            self.lower_body_target_y = input.look_y;
            self.upper_body_y = input.look_y - self.lower_body_y;
        }

        root_local.rotation = Quat::from_axis_angle(Vec3::Y, self.lower_body_y);

        let bullet_point_global = global_transforms
            .get(self.proc_targets.bullet_point)
            .unwrap();
        let spine1_global = global_transforms.get(self.proc_targets.spine1).unwrap();
        let mut spine1_local = transforms.get_mut(self.proc_targets.spine1).unwrap();
        let root_global = global_transforms.get(root_entity).unwrap();

        if !self.is_sprinting {
            let anim = player.animation(self.anims.get(AnimationName::Sprint));
            let max_angle = (self.config.sprint_reaim_max_angle)(anim);

            rotate_spine_to_x(
                root_global,
                bullet_point_global,
                spine1_global,
                &mut spine1_local,
                input.look_x,
                self.upper_body_y,
                max_angle,
            );
        } else {
            spine1_local.rotation = spine1_local
                .rotation
                .rotate_towards(Quat::IDENTITY, self.config.spine1_into_sprint_max_angle);
        }
    }
}

/// Fades a number of animations out for a single timestep. Stops animations
/// where the weight is less than `threshold`. Fade done by multiplying the weight
/// by `rate`.
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

/// Fades an animation in for a single timestep. If the animation is not playing,
/// it starts with a weight of `threshold`. Fade is done by multiplying the weight
/// by `rate`.
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
    mut target_gun_x_rotation: f32,
    target_gun_y_rotation: f32,
    max_angle: f32,
) {
    // The maximum angle to rotate per frame to coerce the players spine to mostly be upwards.
    // This is so that the player doesn't point the gun forward by hunching over to one side.
    const UPWARDS_PULL: f32 = 0.1;

    if target_gun_x_rotation > FRAC_PI_2 * 0.9 || target_gun_x_rotation < -FRAC_PI_2 * 0.9 {
        warn!("gun x rotation too large");
        target_gun_x_rotation = target_gun_x_rotation.clamp(-FRAC_PI_2, FRAC_PI_2 * 0.9);
    }

    // Compute target gun forward direction in local space.
    let target_vec = Quat::from_axis_angle(Vec3::Y, target_gun_y_rotation)
        * Quat::from_axis_angle(Vec3::X, target_gun_x_rotation)
        * Vec3::Z;
    // Convert target from player local space to global space.
    let global_target = player_global.rotation() * target_vec;

    // Compute the current forward direction of the bullet_point in global space.
    let current_bullet_forward = bullet_point_global.rotation() * Vec3::Z;

    // Compute the rotation needed to align the current bullet_point forward to the target direction.
    // This is done in spine1's local space.
    let alignment_rotation = Quat::from_rotation_arc(
        spine1_global.rotation().inverse() * current_bullet_forward,
        spine1_global.rotation().inverse() * global_target,
    );

    let spine1_rotation = spine1_local.rotation;
    let up = Quat::from_axis_angle(Vec3::Y, 0.0);
    let target_rotation = spine1_rotation * alignment_rotation;
    let target_pulled_up = target_rotation.rotate_towards(up, UPWARDS_PULL);
    let spine1_rotated = spine1_rotation.rotate_towards(target_pulled_up, max_angle);
    spine1_local.rotation = spine1_rotated;
}
