use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use std::ops::AddAssign;

pub struct EnemyPlugin;

impl Plugin for EnemyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, enemy_update);
    }
}

#[derive(Component)]
#[require(Collider, KinematicCharacterController, RigidBody)]
pub struct Enemy {
    pub state: EnemyState,
}

impl Enemy {
    pub fn new(translation: Vec3) -> impl Bundle {
		let controller = KinematicCharacterController::default();

        (
            Collider::ball(0.5),
            Enemy {
                state: EnemyState::Idle,
            },
			controller,
            RigidBody::KinematicPositionBased,
            Transform::from_translation(translation),
        )
    }
}

pub enum EnemyState {
    Idle,
    Moving(EnemyPath),
    AttackJump,
}

pub struct EnemyPath {
    pub path: Vec<Vec3>,
    pub target_index: usize,
    pub speed: f32,
}

fn enemy_update(
    mut query: Query<(
        &GlobalTransform,
        &mut KinematicCharacterController,
        &mut Enemy,
        Option<&KinematicCharacterControllerOutput>,
    )>,
    time: Res<Time>,
) {
    for (transform, mut controller, mut enemy, output) in query.iter_mut() {
		if output.is_some_and(|a| !a.grounded) {
			let gravity = -9.8 * Vec3::Y * time.delta_secs();
			controller
				.translation
				.get_or_insert(Vec3::ZERO)
				.add_assign(gravity);
		}
        match enemy.state {
            EnemyState::Moving(ref mut path_data) => {
                if path_data.target_index >= path_data.path.len() {
                    continue; // Path complete
                }

                // Get current position and target
                let current_position = transform.translation();
                let target_position = path_data.path[path_data.target_index];

                // Calculate direction and move
                let direction = (target_position - current_position).normalize_or_zero();
                let step = path_data.speed * time.delta_secs();
                controller
                    .translation
                    .get_or_insert(Vec3::ZERO)
                    .add_assign(direction * step);
				dbg!(controller.translation);

                // Check if target is reached
                if current_position.distance(target_position) < 1.0 {
                    path_data.target_index += 1; // Move to the next target point
                }
            }
            EnemyState::Idle => {}
            EnemyState::AttackJump => {
                if output.is_some_and(|v| v.grounded) {
                    enemy.state = EnemyState::Idle;
                }
            }
        }
    }
}
