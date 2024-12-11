use bevy::{
    prelude::*,
    window::{CursorGrabMode, PrimaryWindow},
};

pub fn toggle_cursor_grab_with_esc(
    keys: Res<ButtonInput<KeyCode>>,
    mut q_windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        let mut primary_window = q_windows.single_mut();
        primary_window.cursor_options.visible = !primary_window.cursor_options.visible;
        primary_window.cursor_options.grab_mode = if primary_window.cursor_options.visible {
            CursorGrabMode::None
        } else {
            CursorGrabMode::Locked
        };
    }
}

pub mod freecam {
    use bevy::{
        input::mouse::MouseMotion,
        prelude::*,
        window::{CursorGrabMode, PrimaryWindow},
    };

    pub struct FreeCameraPlugin;

    impl Plugin for FreeCameraPlugin {
        fn build(&self, app: &mut App) {
            app.add_systems(Update, (free_camera_movement, mouse_look));
        }
    }

    #[derive(Component)]
    pub struct FreeCamera {
        pub speed: f32,
        pub movement_enabled: bool,
    }

    impl FreeCamera {
        pub fn new(speed: f32) -> Self {
            Self {
                speed,
                movement_enabled: true,
            }
        }
    }

    // Free camera system
    fn free_camera_movement(
        time: Res<Time>,
        keys: Res<ButtonInput<KeyCode>>,
        mut query: Query<(&mut Transform, &mut FreeCamera)>,
    ) {
        for (mut transform, camera) in query.iter_mut() {
            if !camera.movement_enabled {
                return;
            }
            let forward = transform.rotation.mul_vec3(Vec3::new(0.0, 0.0, -1.0));
            let right = transform.rotation.mul_vec3(Vec3::new(1.0, 0.0, 0.0));

            let mut wasd_velocity = Vec3::ZERO;
            let mut vertical = 0.0;

            if keys.pressed(KeyCode::KeyW) {
                wasd_velocity += forward;
            }
            if keys.pressed(KeyCode::KeyS) {
                wasd_velocity -= forward;
            }
            if keys.pressed(KeyCode::KeyA) {
                wasd_velocity -= right;
            }
            if keys.pressed(KeyCode::KeyD) {
                wasd_velocity += right;
            }
            if keys.pressed(KeyCode::Space) {
                vertical += 1.0;
            }
            if keys.pressed(KeyCode::ShiftLeft) {
                vertical -= 1.0;
            }

            wasd_velocity.y = 0.0;
            wasd_velocity = wasd_velocity.normalize_or_zero();
            wasd_velocity.y = vertical;
            transform.translation += wasd_velocity * time.delta_secs() * camera.speed;
        }
    }

    /// Rotates the player based on mouse movement.
    fn mouse_look(
        mut mouse_motion: EventReader<MouseMotion>,
        mut camera: Query<&mut Transform, With<FreeCamera>>,
        q_windows: Query<&Window, With<PrimaryWindow>>,
    ) {
        let primary_window = q_windows.single();
        if primary_window.cursor_options.grab_mode != CursorGrabMode::Locked {
            return;
        }
        let Ok(mut camera) = camera.get_single_mut() else {
            return;
        };
        for motion in mouse_motion.read() {
            let yaw = -motion.delta.x * 0.003;
            let pitch = -motion.delta.y * 0.002;
            camera.rotate_y(yaw);
            camera.rotate_local_x(pitch);
        }
    }
}

/// Gets all decendants recursivley, including `entity`.
pub fn get_all_descendants(entity: Entity, children: &Query<&Children>) -> Vec<Entity> {
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
pub fn map_query<T: Component + Clone>(entites: Vec<Entity>, query: &Query<&T>) -> Vec<T> {
    entites
        .into_iter()
        .flat_map(|v| query.get(v).ok())
        .cloned()
        .collect::<Vec<_>>()
}

pub fn find_upwards<'a, T: Component>(
    entity: Entity,
    parents: &Query<&Parent>,
    component: &'a Query<&T>,
) -> Option<&'a T> {
    let mut search = entity;
    while let Ok(parent) = parents.get(search) {
        if let Ok(comp) = component.get(parent.get()) {
            return Some(comp);
        };
        search = parent.get();
    }
    return None;
}

/// Samples from the element in the array corresponding to the most aligned cardinal direction (forward, back, left, right) based on a Vec2.
pub fn sample_cardinal<T>(array: &[T; 4], direction: Vec2) -> &T {
    // Normalize the direction vector to handle non-unit vectors
    let normalized_dir = direction.normalize_or_zero();
    // Define the cardinal directions as unit vectors
    let cardinal_directions = [
        Vec2::new(0.0, 1.0),  // Forward
        Vec2::new(0.0, -1.0), // Back
        Vec2::new(-1.0, 0.0), // Left
        Vec2::new(1.0, 0.0),  // Right
    ];
    // Find the index of the most aligned cardinal direction
    let (max_index, _) = cardinal_directions
        .iter()
        .enumerate()
        .map(|(i, &cardinal)| (i, normalized_dir.dot(cardinal)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    &array[max_index]
}

/// Constructs a unit vector (or zero vector) based on 4 booleans: [forward, back, left, right].
pub fn unit_vector_from_bools(forward: bool, back: bool, left: bool, right: bool) -> Vec2 {
    let mut vec = Vec2::ZERO;
    if forward {
        vec += Vec2::new(0.0, 1.0);
    }
    if back {
        vec += Vec2::new(0.0, -1.0);
    }
    if left {
        vec += Vec2::new(-1.0, 0.0);
    }
    if right {
        vec += Vec2::new(1.0, 0.0);
    }
    vec.normalize_or_zero()
}

/// Recursively searches for a child entity by a path of names, starting from the given root entity.
/// Returns the child entity if found, or `None` if the path is invalid/entity cannot be found.
pub fn find_child_by_path(
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

pub fn find_child_with_name(
    root: Entity,
    name: &str,
    children: &Query<&Children>,
    names: &Query<&Name>,
) -> Option<Entity> {
    if let Ok(n) = names.get(root) {
        if n.as_str() == name {
            return Some(root);
        }
    }

    if let Ok(c) = children.get(root) {
        for child in c {
            if let Some(entity) = find_child_with_name(*child, name, children, names) {
                return Some(entity);
            }
        }
    }

    return None;
}
