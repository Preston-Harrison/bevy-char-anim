use bevy::animation::ActiveAnimation;
use bevy::prelude::*;

use crate::anim::{AnimationName, PlayerAnimations};
use crate::utils::*;

pub struct PlayerAnimationInput {
    /// +Y is forward
    pub local_movement_direction: Vec2,

    pub just_jumped: bool,
    pub is_grounded: bool,
}

#[derive(Component)]
pub struct PlayerAnimationState {
    anims: PlayerAnimations,
    lower_body: LowerBodyState,
}

impl PlayerAnimationState {
    pub fn new(anims: PlayerAnimations) -> Self {
        Self {
            anims,
            lower_body: LowerBodyState::Idle,
        }
    }
}

impl PlayerAnimationState {
    pub fn transition(&mut self, input: &PlayerAnimationInput, player: &AnimationPlayer) {
        let is_finished = |anim| player.animation(anim).unwrap().is_finished();

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
            _ if input.just_jumped => {
                info!("jumped");
                LowerBodyState::Jump
            },
            _ if input.local_movement_direction.length() < 0.1 => {
                    LowerBodyState::Idle
            }
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

    pub fn update_player(&self, player: &mut AnimationPlayer) {
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

        let rate = 0.9;
        let threshold = 0.01;

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
